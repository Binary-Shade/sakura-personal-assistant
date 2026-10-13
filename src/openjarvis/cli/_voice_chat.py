"""Voice input/output helpers for the interactive chat session."""

from __future__ import annotations

from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
import queue
import re
import threading
import os
import wave
from typing import Any, Optional

from rich.markup import escape

VOICE_EXIT = object()
DEFAULT_WAKE_PHRASE = "hey sakura"
DEFAULT_WAKE_IDLE_TIMEOUT_SECONDS = 300.0
_WAKE_DEBUG_ENV = "SAKURA_WAKE_DEBUG"
_WAKE_WARNED: set[str] = set()
_WAKE_SAMPLE_MATCHER: "_WakeSampleMatcher | None" = None

# Backend selection is shared with the API server and lives in
# openjarvis.speech._tts_discovery. Importing that at module level would pull
# the whole speech stack -- numpy included -- into every `jarvis` invocation,
# which tests/cli/test_cli.py guards against, so these aliases resolve lazily.
_LAZY_TTS_NAMES = {
    "_TTS_BACKEND_ORDER": "TTS_BACKEND_ORDER",
    "_BACKEND_DEFAULT_VOICE": "BACKEND_DEFAULT_VOICE",
}


def __getattr__(name: str) -> Any:
    target = _LAZY_TTS_NAMES.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from openjarvis.speech import _tts_discovery

    return getattr(_tts_discovery, target)


def _terminal_safe_text(value: object) -> str:
    """Remove terminal controls and escape Rich markup from dynamic text."""
    printable = "".join(
        char for char in str(value) if char in ("\n", "\t") or char.isprintable()
    )
    return escape(printable)


class VoiceSession:
    """Cache healthy speech backends for one interactive chat session."""

    def __init__(self, config: object | None = None) -> None:
        self._config = config
        self._stt_resolved = False
        self._stt_backend: Any = None
        self._tts_backend: Any = None
        self._tts_attempted: set[str] = set()
        self._voice_prefs: tuple[str, str, float] | None = None
        self._voice_warned: set[str] = set()
        self._tts_warmup_started = False

    def get_stt_backend(self) -> Any:
        """Resolve and health-check STT once, then reuse the loaded backend."""
        if not self._stt_resolved:
            from openjarvis.core.config import load_config
            from openjarvis.speech._discovery import get_speech_backend

            config = self._config if self._config is not None else load_config()
            self._stt_backend = get_speech_backend(config)
            self._stt_resolved = True
        return self._stt_backend

    def get_tts_backend(self) -> Any:
        """Return a cached healthy TTS backend, falling through once per key."""
        if self._tts_backend is not None:
            return self._tts_backend

        from openjarvis.speech._tts_discovery import get_tts_backend

        preferred, _, _ = self.get_voice_preferences()
        self._tts_backend = get_tts_backend(preferred, attempted=self._tts_attempted)
        return self._tts_backend

    def get_voice_preferences(self) -> tuple[str, str, float]:
        """Resolve configured (tts_backend, voice_id, speed), cached per session."""
        if self._voice_prefs is None:
            from openjarvis.core.config import load_config
            from openjarvis.speech._tts_discovery import voice_preferences

            config = self._config if self._config is not None else load_config()
            self._voice_prefs = voice_preferences(config)
        return self._voice_prefs

    def get_stt_language(self) -> str | None:
        """Return the configured STT language, defaulting voice mode to English."""
        from openjarvis.core.config import load_config

        config = self._config if self._config is not None else load_config()
        speech = getattr(config, "speech", None)
        language = str(getattr(speech, "language", "") or "").strip()
        return language or "en"

    def voice_for_backend(self, backend: Any, console: Any = None) -> tuple[str, float]:
        """Return the voice ID valid for ``backend``, plus the configured speed.

        Voice IDs are not portable across backends, so the configured
        ``speech.voice_id`` is honored only when ``backend`` is the configured
        ``speech.tts_backend``. Otherwise the fallback backend's own default is
        used and the substitution is reported once per session.
        """
        want_backend, voice_id, speed = self.get_voice_preferences()
        active = getattr(backend, "backend_id", "") or ""
        if active == want_backend:
            return voice_id, speed

        from openjarvis.speech._tts_discovery import default_voice_for

        substitute = default_voice_for(active)
        if console is not None and active not in self._voice_warned:
            self._voice_warned.add(active)
            console.print(
                f"[dim yellow]Voice: {want_backend!r} unavailable — using "
                f"{active!r} with its default voice "
                f"({substitute or 'backend default'}).[/dim yellow]"
            )
        return substitute, speed

    def discard_tts_backend(self) -> None:
        """Forget a backend that failed synthesis and allow the next fallback."""
        self._tts_backend = None

    def prewarm_tts(self, console: Any) -> None:
        """Start loading the configured TTS model before the first response."""
        if self._tts_warmup_started:
            return
        self._tts_warmup_started = True

        def _warm() -> None:
            try:
                synthesize_speech("Ready.", console, self)
            except Exception:
                # Warmup is only an optimization; real synthesis still reports
                # backend failures when the user needs audio.
                pass

        threading.Thread(
            target=_warm,
            name="sakura-tts-warmup",
            daemon=True,
        ).start()


def read_voice_input(console: Any, session: VoiceSession) -> Optional[str] | object:
    """Accept a typed command/message, or record after an empty submission."""
    try:
        typed = input("You> [type, or press Enter to speak] ")
    except (EOFError, KeyboardInterrupt):
        return VOICE_EXIT
    typed = typed.strip()
    return typed if typed else record_voice(console, session)


def read_wake_voice_input(
    console: Any,
    session: VoiceSession,
    *,
    wake_phrase: str = DEFAULT_WAKE_PHRASE,
) -> Optional[str] | object:
    """Wait for a wake phrase, then record and transcribe the user's command."""
    try:
        typed = input("You> [type, or Enter for wake mode] ")
    except (EOFError, KeyboardInterrupt):
        return VOICE_EXIT
    typed = typed.strip()
    if typed:
        return typed
    return listen_for_wake_phrase(console, session, wake_phrase=wake_phrase)


def listen_for_wake_phrase(
    console: Any,
    session: VoiceSession,
    *,
    wake_phrase: str = DEFAULT_WAKE_PHRASE,
) -> Optional[str] | object:
    """Listen until the configured wake phrase is heard, then capture a turn."""
    backend = session.get_stt_backend()
    if backend is None:
        console.print(
            "[red]No speech-to-text backend available. Wake mode needs STT.[/red]"
        )
        return VOICE_EXIT

    console.print(
        f"[dim cyan]Wake mode: say '{_terminal_safe_text(wake_phrase)}'. "
        "Press Ctrl-C to exit.[/dim cyan]"
    )
    try:
        while True:
            if _listen_with_openwakeword(console):
                console.print("[dim cyan]Wake word detected. Listening…[/dim cyan]")
                return record_voice(console, session)

            if _listen_with_porcupine(console):
                console.print("[dim cyan]Wake word detected. Listening…[/dim cyan]")
                return record_voice(console, session)

            wake_audio = _record_wake_candidate(console)
            if wake_audio is None:
                continue

            if _wake_samples_match(console, wake_audio):
                console.print("[dim cyan]Wake word detected. Listening…[/dim cyan]")
                return record_voice(console, session)

            try:
                result = backend.transcribe(
                    wake_audio,
                    format="wav",
                    language=session.get_stt_language(),
                )
                transcript = result.text.strip()
            except Exception as exc:
                console.print(
                    f"[dim yellow]Wake transcription error: "
                    f"{_terminal_safe_text(exc)}[/dim yellow]"
                )
                _wake_log(f"transcription_error: {exc}")
                continue

            command = _command_after_wake_phrase(transcript, wake_phrase)
            _wake_log(
                f"heard={transcript!r} lang={getattr(result, 'language', None)!r} "
                f"conf={getattr(result, 'confidence', None)!r} "
                f"matched={command is not None}"
            )
            if os.environ.get(_WAKE_DEBUG_ENV) == "1":
                status = "matched" if command is not None else "ignored"
                console.print(
                    f"[dim]Wake debug ({status}): "
                    f"{_terminal_safe_text(transcript or '<empty>')}[/dim]"
                )
            if command is None:
                continue

            console.print("[dim cyan]Wake word detected. Listening…[/dim cyan]")
            if command:
                console.print(
                    f"[bold]You (voice):[/bold] {_terminal_safe_text(command)}"
                )
                return command
            return record_voice(console, session)
    except KeyboardInterrupt:
        return VOICE_EXIT


def _wake_engine_enabled(engine_name: str) -> bool:
    configured = os.environ.get("SAKURA_WAKE_ENGINE", "auto").strip().lower()
    if engine_name == "porcupine":
        return configured == "porcupine"
    if engine_name == "samples":
        return configured in ("", "auto", "samples")
    return configured in ("", "auto", engine_name)


def _warn_wake_once(console: Any, key: str, message: str) -> None:
    if key in _WAKE_WARNED:
        return
    _WAKE_WARNED.add(key)
    console.print(f"[dim yellow]{message}[/dim yellow]")


def _listen_with_porcupine(console: Any) -> bool:
    """Return True when Picovoice Porcupine detects the configured wake word.

    This backend is not used by default because it requires Picovoice licensing.
    To opt in manually, create a Linux .ppn model in Picovoice Console and set:

      SAKURA_WAKE_ENGINE=porcupine
      SAKURA_PICOVOICE_ACCESS_KEY=<key>
      SAKURA_PORCUPINE_KEYWORD_PATH=/path/to/hey_sakura_linux.ppn

    Built-in Porcupine keywords can be tested with SAKURA_PORCUPINE_KEYWORD.
    """
    if not _wake_engine_enabled("porcupine"):
        return False

    access_key = os.environ.get("SAKURA_PICOVOICE_ACCESS_KEY", "").strip()
    keyword_path = os.environ.get("SAKURA_PORCUPINE_KEYWORD_PATH", "").strip()
    keyword = os.environ.get("SAKURA_PORCUPINE_KEYWORD", "").strip()
    if not access_key or not (keyword_path or keyword):
        _warn_wake_once(
            console,
            "porcupine-config",
            "Porcupine wake mode needs SAKURA_PICOVOICE_ACCESS_KEY and "
            "SAKURA_PORCUPINE_KEYWORD_PATH.",
        )
        return False

    try:
        import numpy as np
        import pvporcupine
        import sounddevice as sd
    except ImportError as exc:
        _warn_wake_once(
            console,
            "porcupine-import",
            "pvporcupine is not installed; falling back to the next wake detector. "
            f"Original error: {_terminal_safe_text(exc)}",
        )
        return False

    sensitivity = float(os.environ.get("SAKURA_PORCUPINE_SENSITIVITY", "0.72"))
    porcupine = None
    try:
        if keyword_path:
            porcupine = pvporcupine.create(
                access_key=access_key,
                keyword_paths=[keyword_path],
                sensitivities=[sensitivity],
            )
        else:
            porcupine = pvporcupine.create(
                access_key=access_key,
                keywords=[keyword],
                sensitivities=[sensitivity],
            )
    except Exception as exc:
        _warn_wake_once(
            console,
            "porcupine-load",
            "Failed to load Porcupine wake model; falling back to the next "
            f"wake detector. Original error: {_terminal_safe_text(exc)}",
        )
        return False

    try:
        with sd.RawInputStream(
            samplerate=porcupine.sample_rate,
            channels=1,
            dtype="int16",
            blocksize=porcupine.frame_length,
        ) as stream:
            while True:
                raw, _ = stream.read(porcupine.frame_length)
                pcm = np.frombuffer(bytes(raw), dtype=np.int16)
                if porcupine.process(pcm) >= 0:
                    _wake_log(
                        "porcupine_detected "
                        f"sensitivity={sensitivity:.2f} "
                        f"keyword_path={bool(keyword_path)} keyword={keyword!r}"
                    )
                    return True
    finally:
        porcupine.delete()


def _listen_with_openwakeword(console: Any) -> bool:
    """Return True when a configured openWakeWord model fires.

    Set ``SAKURA_WAKE_MODEL_PATH`` to a custom openWakeWord model for the
    phrase "hey sakura". Without that model, the caller falls back to STT phrase
    detection so wake mode still works during development.
    """
    if not _wake_engine_enabled("openwakeword"):
        return False

    model_path = os.environ.get("SAKURA_WAKE_MODEL_PATH", "").strip()
    if not model_path:
        if os.environ.get("SAKURA_WAKE_ENGINE", "auto").strip().lower() == "openwakeword":
            _warn_wake_once(
                console,
                "openwakeword-config",
                "openWakeWord wake mode needs SAKURA_WAKE_MODEL_PATH for a "
                "custom 'hey sakura' model.",
            )
        return False

    try:
        import numpy as np
        import sounddevice as sd
        from openwakeword.model import Model
    except ImportError as exc:
        console.print(
            "[dim yellow]openWakeWord is not installed; falling back to "
            f"STT wake detection. Original error: {_terminal_safe_text(exc)}"
            "[/dim yellow]"
        )
        return False

    threshold = float(os.environ.get("SAKURA_WAKE_THRESHOLD", "0.42"))
    debug_scores = os.environ.get(_WAKE_DEBUG_ENV) == "1"
    frame_size = int(os.environ.get("SAKURA_WAKE_FRAME_SIZE", "1280"))
    try:
        model = Model(wakeword_model_paths=[model_path])
    except Exception as exc:
        console.print(
            "[dim yellow]Failed to load openWakeWord model; falling back to "
            f"STT wake detection. Original error: {_terminal_safe_text(exc)}"
            "[/dim yellow]"
        )
        return False

    with sd.InputStream(
        samplerate=16000,
        channels=1,
        dtype="int16",
        blocksize=frame_size,
    ) as stream:
        while True:
            audio, _ = stream.read(frame_size)
            frame = np.asarray(audio).reshape(-1)
            prediction = model.predict(frame)
            score = max(float(value) for value in prediction.values())
            if debug_scores and score >= threshold * 0.65:
                _wake_log(
                    f"openwakeword_score score={score:.3f} threshold={threshold:.3f}"
                )
            if score >= threshold:
                _wake_log(
                    f"openwakeword_detected score={score:.3f} "
                    f"threshold={threshold:.3f}"
                )
                return True


def _record_wake_candidate(console: Any) -> bytes | None:
    from openjarvis.speech.voice_io import (
        record_until_silence,
        wav_duration_seconds,
        wav_rms,
    )

    try:
        audio = record_until_silence(
            silence_seconds=0.65,
            startup_silence_seconds=30.0,
            max_seconds=3.2,
        )
        duration = wav_duration_seconds(audio)
        rms = wav_rms(audio)
        if duration < 0.35 or rms < 80:
            _wake_log(f"skipped_audio duration={duration:.2f} rms={rms:.1f}")
            return None
        return audio
    except KeyboardInterrupt:
        raise
    except Exception as exc:
        console.print(f"[red]Mic error: {_terminal_safe_text(exc)}[/red]")
        return None


class _WakeSampleMatcher:
    """Query-by-example wake matcher using the user's recorded samples."""

    def __init__(self, samples_dir: Path) -> None:
        import numpy as np
        from openwakeword.utils import AudioFeatures

        self._np = np
        self._features = AudioFeatures()
        self._positive = self._load_templates(samples_dir / "positive")
        self._negative = self._load_templates(samples_dir / "negative")

    @property
    def ready(self) -> bool:
        return len(self._positive) >= 3

    def score_wav(self, audio: bytes) -> tuple[float, float | None, float]:
        candidate = self._embed_wav_bytes(audio)
        return self.score_embedding(candidate)

    def score_path(self, path: Path, *, exclude_self: bool = False) -> tuple[float, float | None, float]:
        candidate = self._embed_wav_file(path)
        return self.score_embedding(
            candidate,
            exclude_path=path if exclude_self else None,
        )

    def score_embedding(
        self,
        candidate: object,
        *,
        exclude_path: Path | None = None,
    ) -> tuple[float, float | None, float]:
        positives = [
            template
            for path, template in self._positive
            if exclude_path is None or path != exclude_path
        ]
        if not positives:
            positives = [template for _, template in self._positive]
        positive_score = max(
            self._sequence_similarity(candidate, template)
            for template in positives
        )
        negative_score = (
            max(
                self._sequence_similarity(candidate, template)
                for _, template in self._negative
            )
            if self._negative
            else None
        )
        margin = (
            positive_score - negative_score
            if negative_score is not None
            else positive_score
        )
        return positive_score, negative_score, margin

    def _load_templates(self, directory: Path) -> list[object]:
        templates = []
        if not directory.exists():
            return templates
        for path in sorted(directory.glob("*.wav")):
            try:
                templates.append((path, self._embed_wav_file(path)))
            except Exception:
                _wake_log(f"sample_load_failed path={path}")
        return templates

    def _embed_wav_file(self, path: Path) -> object:
        with wave.open(str(path), "rb") as handle:
            audio = self._np.frombuffer(
                handle.readframes(handle.getnframes()),
                dtype=self._np.int16,
            )
        return self._embed_pcm(audio)

    def _embed_wav_bytes(self, audio: bytes) -> object:
        import io

        with wave.open(io.BytesIO(audio), "rb") as handle:
            pcm = self._np.frombuffer(
                handle.readframes(handle.getnframes()),
                dtype=self._np.int16,
            )
        return self._embed_pcm(pcm)

    def _embed_pcm(self, pcm: object) -> object:
        embeddings = self._features._get_embeddings(pcm)
        if embeddings.ndim == 1:
            embeddings = embeddings[None, :]
        norms = self._np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings / self._np.maximum(norms, 1e-8)

    def _sequence_similarity(self, left: object, right: object) -> float:
        # Dynamic-time-warping over cosine distance. Clips can differ in speed
        # and length, so sequence alignment is more reliable than a flat mean.
        n, m = left.shape[0], right.shape[0]
        costs = 1.0 - self._np.clip(left @ right.T, -1.0, 1.0)
        dp = self._np.full((n + 1, m + 1), self._np.inf, dtype=self._np.float32)
        dp[0, 0] = 0.0
        for row in range(1, n + 1):
            for col in range(1, m + 1):
                dp[row, col] = costs[row - 1, col - 1] + min(
                    dp[row - 1, col],
                    dp[row, col - 1],
                    dp[row - 1, col - 1],
                )
        path_len = max(n, m)
        return float(1.0 - (dp[n, m] / path_len))


def _wake_samples_match(console: Any, audio: bytes) -> bool:
    if not _wake_engine_enabled("samples"):
        return False

    global _WAKE_SAMPLE_MATCHER
    samples_dir = Path(
        os.environ.get(
            "SAKURA_WAKE_SAMPLES_DIR",
            str(Path.home() / ".openjarvis" / "wake_samples"),
        )
    )
    try:
        if _WAKE_SAMPLE_MATCHER is None:
            matcher = _WakeSampleMatcher(samples_dir)
            if not matcher.ready:
                return False
            _WAKE_SAMPLE_MATCHER = matcher
        positive, negative, margin = _WAKE_SAMPLE_MATCHER.score_wav(audio)
    except Exception as exc:
        _warn_wake_once(
            console,
            "sample-matcher",
            "Wake sample matcher is unavailable; falling back to STT wake "
            f"detection. Original error: {_terminal_safe_text(exc)}",
        )
        return False

    threshold = float(os.environ.get("SAKURA_WAKE_SAMPLE_THRESHOLD", "0.82"))
    min_margin = float(os.environ.get("SAKURA_WAKE_SAMPLE_MARGIN", "0.03"))
    matched = positive >= threshold and margin >= min_margin
    if os.environ.get(_WAKE_DEBUG_ENV) == "1" or matched:
        _wake_log(
            f"sample_match positive={positive:.3f} "
            f"negative={negative if negative is not None else 'none'} "
            f"margin={margin:.3f} threshold={threshold:.3f} "
            f"min_margin={min_margin:.3f} matched={matched}"
        )
    return matched


def _command_after_wake_phrase(
    transcript: str,
    wake_phrase: str,
) -> str | None:
    from openjarvis.speech.voice_io import normalize_wake_text

    normalized = normalize_wake_text(transcript)
    wake = normalize_wake_text(wake_phrase)
    if not normalized or wake not in normalized:
        return _fuzzy_command_after_wake_phrase(normalized, wake)

    wake_end = normalized.find(wake) + len(wake)
    remainder = normalized[wake_end:].strip()
    return remainder


def _fuzzy_command_after_wake_phrase(
    normalized: str,
    wake: str,
) -> str | None:
    tokens = normalized.split()
    wake_tokens = wake.split()
    if not tokens or not wake_tokens:
        return None

    target_name = wake_tokens[-1]
    greeting_tokens = {"hey", "hi", "hello", "okay", "ok"}
    variants = {
        target_name,
        "sakura",
        "sakra",
        "sacra",
        "sacura",
        "sakura",
        "saccura",
        "sakkura",
        "sakura",
        "shakura",
        "sakhura",
        "sakuraa",
        "sabura",
        "sabra",
        "sakula",
        "sacula",
        "sanggura",
        "sangura",
        "sankura",
    }

    joined = " ".join(tokens)
    joined_compact = "".join(tokens)
    compact_variants = {variant.replace(" ", "") for variant in variants}
    for variant in compact_variants:
        if variant and variant in joined_compact:
            start = joined_compact.find(variant) + len(variant)
            # If matching a compacted phrase like "sakulasearch", the original
            # token positions are ambiguous. Wake-only still moves into command
            # capture; otherwise the normal token loop below extracts commands.
            if start >= len(joined_compact) - 2 or len(tokens) <= 4:
                return ""

    for index, token in enumerate(tokens):
        name_score = max(
            SequenceMatcher(None, token, variant).ratio() for variant in variants
        )
        previous = tokens[index - 1] if index > 0 else ""
        has_greeting = previous in greeting_tokens
        starts_with_name = index == 0 and name_score >= 0.72
        greeted_name = has_greeting and name_score >= 0.62
        if starts_with_name or greeted_name:
            return " ".join(tokens[index + 1 :]).strip()

    # Very short wake-only attempts are often transcribed as just the assistant
    # name. Accept those so Sakura moves into command capture instead of
    # silently ignoring the user.
    if len(tokens) <= 3:
        for token in tokens:
            if SequenceMatcher(None, token, target_name).ratio() >= 0.68:
                return ""
    return None


def _wake_log(message: str) -> None:
    try:
        path = Path.home() / ".openjarvis" / ".state" / "wake.log"
        path.parent.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().isoformat(timespec="seconds")
        with path.open("a", encoding="utf-8") as handle:
            handle.write(f"{timestamp} {message}\n")
    except Exception:
        pass


def record_voice(
    console: Any,
    session: VoiceSession | None = None,
    *,
    startup_silence_seconds: float | None = None,
    listening_label: str | None = None,
) -> Optional[str] | object:
    """Record from mic, transcribe, and return text or a loop sentinel."""
    from openjarvis.speech.voice_io import record_until_silence, wav_duration_seconds

    active_session = session or VoiceSession()
    backend = active_session.get_stt_backend()
    if backend is None:
        console.print(
            "[red]No speech-to-text backend available. "
            "Install the voice dependencies with: "
            "install Sakura speech dependencies, or configure a healthy "
            "OpenAI/Deepgram backend.[/red]"
        )
        return VOICE_EXIT

    label = listening_label or "Listening… (speak now, stops on silence)"
    console.print(f"[dim cyan]{label}[/dim cyan]")
    try:
        record_kwargs: dict[str, float] = {}
        if startup_silence_seconds is not None:
            record_kwargs["startup_silence_seconds"] = startup_silence_seconds
        audio_bytes = record_until_silence(**record_kwargs)
        if wav_duration_seconds(audio_bytes) <= 0:
            console.print("[dim]Nothing heard — try again.[/dim]")
            return None
    except KeyboardInterrupt:
        return VOICE_EXIT
    except Exception as exc:
        # PortAudio failures are regular Exceptions (and can surface as
        # OSError), so report them without letting terminal control sequences
        # through. SystemExit deliberately continues to propagate, while
        # Ctrl-C maps to the chat loop's graceful-exit sentinel above.
        console.print(f"[red]Mic error: {_terminal_safe_text(exc)}[/red]")
        return VOICE_EXIT

    console.print("[dim]Transcribing…[/dim]")
    try:
        result = backend.transcribe(
            audio_bytes,
            format="wav",
            language=active_session.get_stt_language(),
        )
        text = result.text.strip()
        if text:
            console.print(f"[bold]You (voice):[/bold] {_terminal_safe_text(text)}")
            return text
        console.print("[dim]Nothing heard — try again.[/dim]")
        return None
    except Exception as exc:
        console.print(f"[red]Transcription error: {_terminal_safe_text(exc)}[/red]")
        return None


def speak(text: str, console: Any, session: VoiceSession | None = None) -> None:
    """Synthesize and play text, reusing a healthy backend for the session."""
    from openjarvis.speech.voice_io import play_wav

    audio = synthesize_speech(text, console, session)
    if audio is not None:
        audio_bytes, sample_rate = audio
        play_wav(audio_bytes, sample_rate=sample_rate)


def synthesize_speech(
    text: str,
    console: Any,
    session: VoiceSession | None = None,
) -> tuple[bytes, int] | None:
    """Synthesize text and return WAV bytes without blocking on playback."""
    active_session = session or VoiceSession()

    # Resolved before the loop: a bad config value must surface as a config
    # error, not be swallowed by the per-backend fallback handler below.
    try:
        active_session.get_voice_preferences()
    except Exception as exc:
        console.print(f"[red]Invalid speech config: {_terminal_safe_text(exc)}[/red]")
        return None

    while (backend := active_session.get_tts_backend()) is not None:
        try:
            voice_id, speed = active_session.voice_for_backend(backend, console)
            synth_kwargs: dict[str, Any] = {"output_format": "wav", "speed": speed}
            if voice_id:
                synth_kwargs["voice_id"] = voice_id
            result = backend.synthesize(text, **synth_kwargs)
            if result.audio:
                return result.audio, result.sample_rate
            return None
        except Exception as exc:
            console.print(
                f"[dim yellow]Voice backend "
                f"{getattr(backend, 'backend_id', '?')!r} failed: "
                f"{_terminal_safe_text(exc)}[/dim yellow]"
            )
            active_session.discard_tts_backend()

    console.print(
        "[dim yellow]No TTS backend available — install kokoro: "
        "pip install kokoro[/dim yellow]"
    )
    return None


_SENTENCE_BOUNDARY_RE = re.compile(r"([.!?]+[\"')\]]*\s+|\n+)")
_CLAUSE_BOUNDARY_RE = re.compile(r"([,;:][\"')\]]*\s+)")


class SpeechChunker:
    """Incrementally split streamed model text into TTS-friendly chunks."""

    def __init__(
        self,
        *,
        first_chunk_chars: int = 48,
        min_sentence_chars: int = 24,
        min_clause_chars: int = 80,
        force_chars: int = 220,
    ) -> None:
        self._buffer = ""
        self._pending_short = ""
        self._emitted_any = False
        self._first_chunk_chars = first_chunk_chars
        self._min_sentence_chars = min_sentence_chars
        self._min_clause_chars = min_clause_chars
        self._force_chars = force_chars

    def feed(self, text: str) -> list[str]:
        """Return speakable chunks completed by this token fragment."""
        self._buffer += text
        chunks: list[str] = []

        while True:
            sentence_match = _first_match(_SENTENCE_BOUNDARY_RE, self._buffer)
            if sentence_match is not None:
                chunk, self._buffer = _split_at_match(self._buffer, sentence_match)
                if chunk:
                    chunks.extend(self._accept_chunk(chunk, allow_pending=True))
                continue

            if len(self._buffer) >= self._min_clause_chars:
                clause_match = _first_match(_CLAUSE_BOUNDARY_RE, self._buffer)
                if (
                    clause_match is not None
                    and clause_match.end() >= self._min_clause_chars
                ):
                    chunk, self._buffer = _split_at_match(self._buffer, clause_match)
                    if chunk:
                        chunks.extend(self._accept_chunk(chunk))
                    continue

            if not self._emitted_any and len(self._buffer) >= self._first_chunk_chars:
                split_at = self._buffer.rfind(" ", 0, self._first_chunk_chars)
                if split_at < self._min_sentence_chars:
                    split_at = self._first_chunk_chars
                chunk = self._buffer[:split_at].strip()
                self._buffer = self._buffer[split_at:].lstrip()
                if chunk:
                    chunks.extend(self._accept_chunk(chunk))
                continue

            if len(self._buffer) >= self._force_chars:
                split_at = self._buffer.rfind(" ", 0, self._force_chars)
                if split_at < self._min_clause_chars:
                    split_at = self._force_chars
                chunk = self._buffer[:split_at].strip()
                self._buffer = self._buffer[split_at:].lstrip()
                if chunk:
                    chunks.extend(self._accept_chunk(chunk))
                continue

            break

        return chunks

    def flush(self) -> list[str]:
        """Return any remaining buffered text as the final speech chunk."""
        chunk = self._buffer.strip()
        self._buffer = ""
        if self._pending_short:
            chunk = f"{self._pending_short} {chunk}".strip()
            self._pending_short = ""
        return [chunk] if chunk else []

    def _accept_chunk(self, chunk: str, *, allow_pending: bool = False) -> list[str]:
        if self._pending_short:
            chunk = f"{self._pending_short} {chunk}".strip()
            self._pending_short = ""
        if allow_pending and len(chunk) < self._min_sentence_chars:
            self._pending_short = chunk
            return []
        self._emitted_any = True
        return [chunk]


def _first_match(pattern: re.Pattern[str], text: str) -> re.Match[str] | None:
    return next(pattern.finditer(text), None)


def _split_at_match(text: str, match: re.Match[str]) -> tuple[str, str]:
    end = match.end()
    return text[:end].strip(), text[end:].lstrip()


class StreamingSpeaker:
    """Synthesize and play TTS chunks in order while text is still streaming."""

    def __init__(self, console: Any, session: VoiceSession) -> None:
        self._console = console
        self._session = session
        self._text_queue: queue.Queue[str | None] = queue.Queue()
        self._audio_queue: queue.Queue[tuple[bytes, int] | None] = queue.Queue()
        self._stop = threading.Event()
        self._synth_thread = threading.Thread(
            target=self._run_synthesis,
            name="sakura-streaming-tts-synth",
            daemon=True,
        )
        self._play_thread = threading.Thread(
            target=self._run_playback,
            name="sakura-streaming-tts-playback",
            daemon=True,
        )
        self._synth_thread.start()
        self._play_thread.start()

    def submit(self, text: str) -> None:
        chunk = text.strip()
        if chunk and not self._stop.is_set():
            self._text_queue.put(chunk)

    def close(self, *, wait: bool = True) -> None:
        self._text_queue.put(None)
        if wait:
            self._synth_thread.join()
            self._play_thread.join()

    def cancel(self) -> None:
        self._stop.set()
        self._drain_queue(self._text_queue)
        self._drain_queue(self._audio_queue)
        self._text_queue.put(None)
        self._audio_queue.put(None)

    @staticmethod
    def _drain_queue(target: queue.Queue[Any]) -> None:
        try:
            while True:
                target.get_nowait()
        except queue.Empty:
            pass

    def _run_synthesis(self) -> None:
        while not self._stop.is_set():
            text = self._text_queue.get()
            if text is None:
                self._audio_queue.put(None)
                return
            audio = synthesize_speech(text, self._console, self._session)
            if audio is not None and not self._stop.is_set():
                self._audio_queue.put(audio)

    def _run_playback(self) -> None:
        from openjarvis.speech.voice_io import play_wav

        while not self._stop.is_set():
            audio = self._audio_queue.get()
            if audio is None:
                return
            audio_bytes, sample_rate = audio
            play_wav(audio_bytes, sample_rate=sample_rate)


__all__ = [
    "VOICE_EXIT",
    "SpeechChunker",
    "StreamingSpeaker",
    "VoiceSession",
    "DEFAULT_WAKE_PHRASE",
    "DEFAULT_WAKE_IDLE_TIMEOUT_SECONDS",
    "listen_for_wake_phrase",
    "read_wake_voice_input",
    "read_voice_input",
    "record_voice",
    "speak",
    "synthesize_speech",
]
