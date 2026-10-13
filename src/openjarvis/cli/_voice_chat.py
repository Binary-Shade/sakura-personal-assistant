"""Voice input/output helpers for the interactive chat session."""

from __future__ import annotations

import queue
import re
import threading
import os
from typing import Any, Optional

from rich.markup import escape

VOICE_EXIT = object()
DEFAULT_WAKE_PHRASE = "hey sakura"

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

            wake_audio = _record_wake_candidate(console)
            if wake_audio is None:
                continue

            try:
                transcript = backend.transcribe(wake_audio, format="wav").text.strip()
            except Exception as exc:
                console.print(
                    f"[dim yellow]Wake transcription error: "
                    f"{_terminal_safe_text(exc)}[/dim yellow]"
                )
                continue

            command = _command_after_wake_phrase(transcript, wake_phrase)
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


def _listen_with_openwakeword(console: Any) -> bool:
    """Return True when a configured openWakeWord model fires.

    Set ``SAKURA_WAKE_MODEL_PATH`` to a custom openWakeWord model for the
    phrase "hey sakura". Without that model, the caller falls back to STT phrase
    detection so wake mode still works during development.
    """
    model_path = os.environ.get("SAKURA_WAKE_MODEL_PATH", "").strip()
    if not model_path:
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

    threshold = float(os.environ.get("SAKURA_WAKE_THRESHOLD", "0.55"))
    frame_size = int(os.environ.get("SAKURA_WAKE_FRAME_SIZE", "1280"))
    try:
        model = Model(wakeword_models=[model_path])
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
            if score >= threshold:
                return True


def _record_wake_candidate(console: Any) -> bytes | None:
    from openjarvis.speech.voice_io import record_until_silence

    try:
        return record_until_silence(
            silence_seconds=0.65,
            startup_silence_seconds=60.0,
            max_seconds=4.0,
        )
    except KeyboardInterrupt:
        raise
    except Exception as exc:
        console.print(f"[red]Mic error: {_terminal_safe_text(exc)}[/red]")
        return None


def _command_after_wake_phrase(
    transcript: str,
    wake_phrase: str,
) -> str | None:
    from openjarvis.speech.voice_io import normalize_wake_text

    normalized = normalize_wake_text(transcript)
    wake = normalize_wake_text(wake_phrase)
    if not normalized or wake not in normalized:
        return None

    wake_end = normalized.find(wake) + len(wake)
    remainder = normalized[wake_end:].strip()
    return remainder


def record_voice(
    console: Any,
    session: VoiceSession | None = None,
) -> Optional[str] | object:
    """Record from mic, transcribe, and return text or a loop sentinel."""
    from openjarvis.speech.voice_io import record_until_silence

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

    console.print("[dim cyan]Listening… (speak now, stops on silence)[/dim cyan]")
    try:
        audio_bytes = record_until_silence()
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
        result = backend.transcribe(audio_bytes, format="wav")
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
    "listen_for_wake_phrase",
    "read_wake_voice_input",
    "read_voice_input",
    "record_voice",
    "speak",
    "synthesize_speech",
]
