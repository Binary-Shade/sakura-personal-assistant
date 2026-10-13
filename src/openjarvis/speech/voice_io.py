"""Microphone recording with silence detection and audio playback helpers."""

from __future__ import annotations

import io
import re
import wave
from typing import Protocol

_SAMPLE_RATE = 16000
_CHANNELS = 1
_CHUNK = 1024
_SILERO_CHUNK = 512
_SILENCE_THRESHOLD = 500  # RMS below this → silence
_SILENCE_SECONDS = 0.9  # seconds of silence before auto-stop
_STARTUP_SILENCE_SECONDS = 5.0  # give up early if speech never begins
_MAX_RECORD_SECONDS = 30  # safety ceiling


class _FrameVAD(Protocol):
    def is_speech(self, data: bytes, sample_rate: int) -> bool:
        """Return True when a PCM frame contains speech."""


def _rms(data: bytes) -> float:
    """Compute RMS amplitude of 16-bit PCM bytes."""
    import struct

    n = len(data) // 2
    if n == 0:
        return 0.0
    shorts = struct.unpack(f"{n}h", data[: n * 2])
    return (sum(s * s for s in shorts) / n) ** 0.5


class _SileroFrameVAD:
    """Tiny frame-level wrapper around silero-vad when it is installed."""

    def __init__(self, *, threshold: float = 0.5) -> None:
        try:
            import numpy as np
            import torch
            from silero_vad import load_silero_vad
        except ImportError as exc:
            raise RuntimeError(
                "silero-vad is not installed. Install with: pip install silero-vad"
            ) from exc

        self._np = np
        self._torch = torch
        self._model = load_silero_vad()
        self._threshold = threshold

    def is_speech(self, data: bytes, sample_rate: int) -> bool:
        audio = (
            self._np.frombuffer(data, dtype=self._np.int16).astype(self._np.float32)
            / 32768.0
        )
        with self._torch.no_grad():
            probability = float(self._model(self._torch.from_numpy(audio), sample_rate))
        return probability >= self._threshold


class _RmsFrameVAD:
    """Fallback VAD for systems without silero-vad."""

    def __init__(self, *, threshold: int) -> None:
        self._threshold = threshold

    def is_speech(self, data: bytes, sample_rate: int) -> bool:
        return _rms(data) > self._threshold


def _get_frame_vad(
    *,
    mode: str,
    silence_threshold: int,
) -> _FrameVAD:
    requested = mode.lower().strip()
    if requested in ("auto", "silero"):
        try:
            return _SileroFrameVAD()
        except RuntimeError:
            if requested == "silero":
                raise
    return _RmsFrameVAD(threshold=silence_threshold)


def normalize_wake_text(text: str) -> str:
    """Normalize transcript text for wake-phrase matching."""
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def record_until_silence(
    *,
    sample_rate: int = _SAMPLE_RATE,
    silence_threshold: int = _SILENCE_THRESHOLD,
    silence_seconds: float = _SILENCE_SECONDS,
    startup_silence_seconds: float = _STARTUP_SILENCE_SECONDS,
    max_seconds: float = _MAX_RECORD_SECONDS,
    vad: str = "auto",
) -> bytes:
    """Record from the default microphone until VAD-confirmed silence.

    Returns raw WAV bytes (16-bit mono).
    Raises RuntimeError if sounddevice is not installed.
    """
    try:
        import sounddevice as sd
    except ImportError:
        raise RuntimeError(
            "sounddevice is required for voice input. "
            "Install with: pip install sounddevice"
        )

    frame_vad = _get_frame_vad(mode=vad, silence_threshold=silence_threshold)

    chunk_size = (
        _SILERO_CHUNK
        if isinstance(frame_vad, _SileroFrameVAD) and sample_rate == 16000
        else _CHUNK
    )
    chunks_per_second = sample_rate / chunk_size
    silence_chunks = int(silence_seconds * chunks_per_second)
    startup_silence_chunks = max(1, int(startup_silence_seconds * chunks_per_second))
    max_chunks = int(max_seconds * chunks_per_second)

    frames: list[bytes] = []
    silence_count = 0
    has_speech = False

    with sd.RawInputStream(
        samplerate=sample_rate,
        channels=_CHANNELS,
        dtype="int16",
        blocksize=chunk_size,
    ) as stream:
        for _ in range(max_chunks):
            raw, _ = stream.read(chunk_size)
            data = bytes(raw)
            frames.append(data)

            if frame_vad.is_speech(data, sample_rate):
                has_speech = True
                silence_count = 0
            elif not has_speech and len(frames) >= startup_silence_chunks:
                break
            elif has_speech:
                silence_count += 1
                if silence_count >= silence_chunks:
                    break

    return _frames_to_wav(frames, sample_rate)


def _frames_to_wav(frames: list[bytes], sample_rate: int) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(_CHANNELS)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(b"".join(frames))
    return buf.getvalue()


def play_wav(audio: bytes, sample_rate: int = 24000) -> None:
    """Play raw WAV bytes through the default output device.

    If the bytes are a valid WAV file, sample rate is read from the header;
    otherwise ``sample_rate`` is used as a fallback.
    """
    try:
        import numpy as np
        import sounddevice as sd
        import soundfile as sf
    except ImportError:
        raise RuntimeError(
            "sounddevice, numpy, and soundfile are required for voice output. "
            "Install with: pip install sounddevice numpy soundfile"
        )

    buf = io.BytesIO(audio)
    try:
        data, sr = sf.read(buf, dtype="float32")
    except Exception:
        # Fall back: treat as raw PCM
        import struct

        n = len(audio) // 2
        data = (
            np.array(struct.unpack(f"{n}h", audio[: n * 2]), dtype="float32") / 32768.0
        )
        sr = sample_rate

    sd.play(data, sr)
    sd.wait()


__all__ = ["normalize_wake_text", "play_wav", "record_until_silence"]
