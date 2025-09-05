
from __future__ import annotations
import math
import socket
import hmac
import hashlib
import struct
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import actions  # must be in the same folder


# ----------------------------- Detection --------------------------------------

@dataclass
class DetectionResult:
    frequency_hz: Optional[float]
    confidence_db: float
    dispatched: bool
    peak_power: float
    note: str = ""


def _parabolic_interpolation(mags: np.ndarray, index: int) -> float:
    """Sub-bin peak interpolation for better frequency resolution."""
    if index <= 0 or index >= len(mags) - 1:
        return float(index)
    alpha, beta, gamma = mags[index - 1], mags[index], mags[index + 1]
    denom = (alpha - 2 * beta + gamma)
    if denom == 0:
        return float(index)
    delta = 0.5 * (alpha - gamma) / denom
    return float(index) + float(delta)


def detect_frequency(samples: np.ndarray, sample_rate: int) -> Tuple[Optional[float], float, float]:
    """
    Estimate dominant frequency using Hann window + rFFT + parabolic interpolation.
    Returns (freq_hz or None, peak_magnitude_linear, confidence_db).
    confidence_db ≈ 20*log10(peak / noise_floor).
    """
    if samples.ndim != 1:
        samples = np.mean(samples, axis=-1)

    # float64 for numerical stability
    x = samples.astype(np.float64, copy=False)
    x -= np.mean(x)

    # Early exit for silence
    rms = math.sqrt(float(np.mean(x * x) + 1e-12))
    if rms < 1e-4:
        return None, 0.0, -120.0

    # Zero-pad to next power of 2
    n = 1 << int(math.ceil(math.log2(len(x))))
    if n > len(x):
        xz = np.zeros(n, dtype=np.float64)
        xz[: len(x)] = x
        x = xz

    # Window + rFFT
    window = np.hanning(len(x))
    mags = np.abs(np.fft.rfft(x * window))
    mags[0] = 0.0  # ignore DC

    peak_idx = int(np.argmax(mags))
    peak_mag = float(mags[peak_idx])
    if peak_mag <= 0.0:
        return None, peak_mag, -120.0

    frac_idx = _parabolic_interpolation(mags, peak_idx)
    freq_hz = float(frac_idx) * sample_rate / len(x)

    # Confidence: exclude ±2 bins around peak to estimate noise floor
    mask = np.ones_like(mags, dtype=bool)
    left = max(0, peak_idx - 2)
    right = min(len(mags), peak_idx + 3)
    mask[left:right] = False
    noise_floor = float(np.mean(mags[mask]) + 1e-12)
    confidence_db = 20.0 * math.log10(max(peak_mag, 1e-12) / noise_floor)

    return freq_hz, peak_mag, confidence_db


def process_buffer(samples: np.ndarray, sample_rate: int) -> DetectionResult:
    """Process one audio buffer: detect frequency and route to an action."""
    freq_hz, peak_mag, conf_db = detect_frequency(samples, sample_rate)
    if freq_hz is None:
        return DetectionResult(None, conf_db, False, peak_mag, "silence/no tone")

    # Delegate decision + debounce to actions.py
    if not hasattr(actions, "route_frequency"):
        raise AttributeError(
            "actions.py must define route_frequency(freq_hz: float, confidence_db: float)"
        )
    dispatched = bool(actions.route_frequency(freq_hz, conf_db))
    return DetectionResult(freq_hz, conf_db, dispatched, peak_mag, "ok")


# --------------------------- ESP32 TCP Server ---------------------------------

HOST = "192.168.137.1"   # Listen on all interfaces
PORT = 12345

# Shared key, must match the ESP32 sketch
SHARED_KEY_HEX = "83e15c0a2b6a0f6f3a040a9f9b21f3c77e2b6d7d6d6e1e4a2b8c1d2e3f405062"
SHARED_KEY = bytes.fromhex(SHARED_KEY_HEX)

AUTH_TIMEOUT = 15.0  # seconds
PRINT_EVERY_N_FRAMES = 10  # throttle console output
FRAME_SAMPLES = 256        # ESP32 sends 256-sample frames
EXPECTED_SR = 16000        # per sketch


def _recv_exact(conn: socket.socket, n: int) -> Optional[bytes]:
    buf = bytearray()
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk:
            return None
        buf.extend(chunk)
    return bytes(buf)


def _send_all(conn: socket.socket, data: bytes) -> bool:
    total = 0
    while total < len(data):
        sent = conn.send(data[total:])
        if sent <= 0:
            return False
        total += sent
    return True


def _auth_handshake(conn: socket.socket, timeout: float = AUTH_TIMEOUT) -> bool:
    conn.settimeout(timeout)

    # 1) Send NONCE (32 bytes)
    nonce = np.random.bytes(32)
    if not _send_all(conn, nonce):
        print("[auth] failed sending nonce")
        return False
    # 2) Receive AUTH1 | dev_len(1) | dev_id | tag(32)
    head = _recv_exact(conn, 6)
    if head is None or head[:5] != b"AUTH1":
        print("[auth] invalid or missing AUTH1 header")
        return False
    dev_len = head[5]
    if dev_len < 1 or dev_len > 64:
        print(f"[auth] invalid dev_len: {dev_len}")
        return False
    dev_id = _recv_exact(conn, dev_len)
    tag = _recv_exact(conn, 32)
    if dev_id is None or tag is None:
        print("[auth] missing dev_id or tag")
        return False

    mac = hmac.new(SHARED_KEY, nonce + dev_id, hashlib.sha256).digest()
    ok = hmac.compare_digest(mac, tag)
    _send_all(conn, b"OK" if ok else b"NO")
    conn.settimeout(None)
    return ok


def _parse_wav_header(hdr44: bytes):
    if len(hdr44) != 44:
        raise ValueError("incomplete WAV header")
    if hdr44[0:4] != b"RIFF" or hdr44[8:12] != b"WAVE":
        raise ValueError("not RIFF/WAVE")
    if hdr44[12:16] != b"fmt ":
        raise ValueError("missing fmt subchunk")

    subchunk1_size, = struct.unpack_from("<I", hdr44, 16)
    audio_format,   = struct.unpack_from("<H", hdr44, 20)
    num_channels,   = struct.unpack_from("<H", hdr44, 22)
    sample_rate,    = struct.unpack_from("<I", hdr44, 24)
    byte_rate,      = struct.unpack_from("<I", hdr44, 28)
    block_align,    = struct.unpack_from("<H", hdr44, 32)
    bits_per_sample,= struct.unpack_from("<H", hdr44, 34)

    if hdr44[36:40] != b"data":
        raise ValueError("missing data subchunk")

    data_size,      = struct.unpack_from("<I", hdr44, 40)
    if audio_format != 1:
        raise ValueError(f"non-PCM format: {audio_format}")

    return {
        "sr": int(sample_rate),
        "channels": int(num_channels),
        "bits": int(bits_per_sample),
        "block_align": int(block_align),
        "data_size_declared": int(data_size),
        "byte_rate": int(byte_rate),
    }


def main():
    if len(SHARED_KEY) != 32:
        raise SystemExit(f"[fatal] SHARED_KEY_HEX invalid: {len(SHARED_KEY)} bytes (expected 32)")

    # Set up TCP server
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    except Exception:
        pass

    try:
        print(f"[bind] {HOST}:{PORT}")
        s.bind((HOST, PORT))
    except OSError as e:
        print(f"[warn] bind failed: {e}; retrying on 0.0.0.0")
        s.bind(("0.0.0.0", PORT))

    s.listen(1)
    print("[ok] Waiting for ESP32 connection…")

    conn, addr = s.accept()
    try:
        try:
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        except Exception as e:
            print(f"[net] TCP_NODELAY not set: {e}")
        print(f"[net] Incoming from {addr}")

        if not _auth_handshake(conn):
            print("[auth] FAILED")
            return
        print("[auth] OK")

        # --- WAV header ---
        hdr = _recv_exact(conn, 44)
        if hdr is None:
            print("[wav] No WAV header received")
            return
        try:
            meta = _parse_wav_header(hdr)
        except Exception as e:
            print(f"[wav] Invalid header: {e}")
            return

        print(f"[wav] sr={meta['sr']} Hz  ch={meta['channels']}  bits={meta['bits']}  block_align={meta['block_align']}")
        if meta["channels"] != 1 or meta["bits"] != 16:
            print("[wav] Only mono 16-bit is supported in this server")
            return
        if meta["sr"] != EXPECTED_SR:
            print(f"[warn] Unexpected sample rate: {meta['sr']} (expected {EXPECTED_SR})")

        # --- Receive and process frames ---
        sr = meta["sr"]
        bytes_per_sample = meta["bits"] // 8
        chan = meta["channels"]
        chunk_bytes = FRAME_SAMPLES * bytes_per_sample * chan

        print("[run] Receiving audio… Ctrl+C to stop.")
        frames = 0
        while True:
            data = _recv_exact(conn, chunk_bytes)
            if data is None:
                print("[net] Client closed connection.")
                break

            # Convert int16 LE -> float32 in [-1, 1]
            x = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0

            # Process one buffer
            result = process_buffer(x, sr)

            # Throttled reporting
            frames += 1
            if frames % PRINT_EVERY_N_FRAMES == 0:
                if result.frequency_hz is None:
                    print(f"[{frames}] ~ silence/no tone (conf={result.confidence_db:5.1f} dB)")
                else:
                    fired = "FIRE" if result.dispatched else "—"
                    print(f"[{frames}] f≈{result.frequency_hz:7.2f} Hz  conf={result.confidence_db:5.1f} dB  {fired}")

    except KeyboardInterrupt:
        print("\n[bye] Interrupted by user.")
    finally:
        try:
            conn.close()
        except Exception:
            pass
        s.close()


if __name__ == "__main__":
    main()
