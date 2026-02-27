"""
Real-time microphone transcription using faster-whisper.
Continuously listens to the mic using Voice Activity Detection (VAD)
to capture natural speech segments without cutting words.

All confirmed text is accumulated in `transcript` (long-term memory).
"""

import sys
import io
import wave
import threading
import time
import numpy as np
import pyaudio
from faster_whisper import WhisperModel

# ── Configuration ──────────────────────────────────────────────
SAMPLE_RATE      = 16000       # 16 kHz – optimal for Whisper
CHANNELS         = 1
SAMPLE_WIDTH     = 2           # 16-bit = 2 bytes
CHUNK_DURATION   = 0.05        # 50ms per read (small for responsiveness)
CHUNK_SIZE       = int(SAMPLE_RATE * CHUNK_DURATION)  # 800 samples

# VAD (Voice Activity Detection) settings
SPEECH_THRESH    = 400         # RMS threshold to consider as speech
SILENCE_DURATION = 0.8         # Seconds of silence to end an utterance
MIN_SPEECH_DUR   = 0.3         # Minimum speech duration to process (avoids clicks)

# Whisper settings
MODEL_SIZE       = "medium" # Options: tiny.en, base.en, small.en, medium.en
DEVICE           = "cpu"
COMPUTE_TYPE     = "int8"      # int8 quantization for fast CPU inference


def load_model() -> WhisperModel:
    """Load the faster-whisper model (downloads on first run)."""
    print(f"  Loading Whisper model '{MODEL_SIZE}' …")
    model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)
    print("  Model loaded.\n")
    return model


def audio_to_wav_bytes(audio_data: bytes) -> bytes:
    """Wrap raw PCM audio data into a WAV byte buffer."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(SAMPLE_WIDTH)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio_data)
    buf.seek(0)
    return buf.read()


def rms(data: bytes) -> float:
    """Calculate the root-mean-square (volume level) of audio data."""
    arr = np.frombuffer(data, dtype=np.int16).astype(np.float64)
    if len(arr) == 0:
        return 0.0
    return float(np.sqrt(np.mean(arr ** 2)))


def print_status(label: str, text: str, color: str = ""):
    """Print a status line, clearing the previous one."""
    codes = {"green": "\033[92m", "yellow": "\033[93m", "cyan": "\033[96m", "dim": "\033[90m", "reset": "\033[0m"}
    c = codes.get(color, "")
    r = codes["reset"]
    sys.stdout.write(f"\r{' ' * 100}\r")  # clear line
    sys.stdout.write(f"  {c}{label}{r}  {text}")
    sys.stdout.flush()


def start_transcription(on_utterance=None) -> str:
    """
    Open the microphone and transcribe speech continuously using VAD.

    Instead of fixed chunks, this listens continuously and detects
    natural speech boundaries (start talking → pause → transcribe).
    Words are never cut mid-sentence.

    Parameters
    ----------
    on_utterance : callable, optional
        Callback invoked when a new utterance is transcribed.
        Signature: on_utterance(text: str, utterance_num: int)
        If provided, the callback handles display; otherwise
        the default terminal output is used (standalone mode).

    Returns
    -------
    transcript : str
        The full accumulated transcript when the user stops with Ctrl+C.
    """
    model = load_model()

    # ── Open mic stream ────────────────────────────────────────
    pa = pyaudio.PyAudio()
    stream = pa.open(
        format=pyaudio.paInt16,
        channels=CHANNELS,
        rate=SAMPLE_RATE,
        input=True,
        frames_per_buffer=CHUNK_SIZE,
    )

    # ── State ──────────────────────────────────────────────────
    transcript: str = ""                # long-term memory
    transcript_lock = threading.Lock()
    is_speaking = False
    speech_frames: list[bytes] = []
    silence_start: float | None = None
    utterance_count = 0

    print("=" * 70)
    print("  🎙  REAL-TIME TRANSCRIPTION")
    print(f"     Model: {MODEL_SIZE}  |  Continuous VAD listening")
    print("     Press Ctrl+C to stop")
    print("=" * 70)
    print()
    if not on_utterance:
        print_status("⏳", "Waiting for speech …", "dim")

    def transcribe_utterance(audio_bytes: bytes, utt_num: int):
        """Transcribe a speech segment in the background."""
        nonlocal transcript
        wav_data = audio_to_wav_bytes(audio_bytes)
        buf = io.BytesIO(wav_data)
        segments, _ = model.transcribe(buf, beam_size=1, language="en")
        text_parts = [seg.text.strip() for seg in segments if seg.text.strip()]

        if text_parts:
            chunk_text = " ".join(text_parts)
            with transcript_lock:
                transcript += chunk_text + " "
                current_transcript = transcript.strip()

            # ── If callback is set, let guardian.py handle display
            if on_utterance:
                on_utterance(chunk_text, utt_num)
            else:
                # ── Standalone mode: show detected text + memory
                sys.stdout.write(f"\r{' ' * 100}\r")
                print(f"\n  ✅  DETECTED #{utt_num}: \033[92m{chunk_text}\033[0m")
                print(f"  📝  MEMORY:   \033[96m{current_transcript}\033[0m")
                print()
                print_status("⏳", "Waiting for speech …", "dim")
        else:
            if not on_utterance:
                print_status("⏳", "Waiting for speech …", "dim")

    try:
        while True:
            data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
            level = rms(data)

            if level > SPEECH_THRESH:
                # ── Speech detected ────────────────────────────
                if not is_speaking:
                    is_speaking = True
                    speech_frames = []
                    silence_start = None
                    print_status("�", "Listening …", "yellow")

                speech_frames.append(data)
                silence_start = None  # reset silence timer

            elif is_speaking:
                # ── Silence after speech ───────────────────────
                speech_frames.append(data)  # keep buffering briefly

                if silence_start is None:
                    silence_start = time.time()

                elapsed_silence = time.time() - silence_start

                if elapsed_silence >= SILENCE_DURATION:
                    # Utterance complete — process it
                    is_speaking = False
                    audio_bytes = b"".join(speech_frames)
                    speech_duration = len(audio_bytes) / (SAMPLE_RATE * SAMPLE_WIDTH)

                    if speech_duration >= MIN_SPEECH_DUR:
                        utterance_count += 1
                        print_status("⚙️ ", "Transcribing …", "cyan")

                        t = threading.Thread(
                            target=transcribe_utterance,
                            args=(audio_bytes, utterance_count),
                        )
                        t.daemon = True
                        t.start()
                    else:
                        print_status("⏳", "Waiting for speech …", "dim")

                    speech_frames = []
                    silence_start = None

    except KeyboardInterrupt:
        # Process any remaining speech
        if speech_frames:
            audio_bytes = b"".join(speech_frames)
            speech_duration = len(audio_bytes) / (SAMPLE_RATE * SAMPLE_WIDTH)
            if speech_duration >= MIN_SPEECH_DUR:
                utterance_count += 1
                print(f"\n\n  ⚙️  Transcribing final utterance …")
                wav_data = audio_to_wav_bytes(audio_bytes)
                buf = io.BytesIO(wav_data)
                segments, _ = model.transcribe(buf, beam_size=1, language="en")
                text_parts = [seg.text.strip() for seg in segments if seg.text.strip()]
                if text_parts:
                    chunk_text = " ".join(text_parts)
                    transcript += chunk_text + " "
                    print(f"  ✅  DETECTED #{utterance_count}: \033[92m{chunk_text}\033[0m")

    finally:
        stream.stop_stream()
        stream.close()
        pa.terminate()

    return transcript.strip()


# ── Entry point ────────────────────────────────────────────────
if __name__ == "__main__":
    full_transcript = start_transcription()

    print("\n" + "=" * 70)
    print("  📝  FULL LONG-TERM MEMORY")
    print("=" * 70)
    print(f"  {full_transcript}")
    print("=" * 70)
