import math
import socket
import hmac
import hashlib
import struct
from typing import Optional
import wave
from datetime import datetime
import numpy as np

# --------------------------- ESP32 TCP Server ---------------------------------

HOST = "192.168.137.1"
PORT = 12345

SHARED_KEY_HEX = "83e15c0a2b6a0f6f3a040a9f9b21f3c77e2b6d7d6d6e1e4a2b8c1d2e3f405062"
SHARED_KEY = bytes.fromhex(SHARED_KEY_HEX)

AUTH_TIMEOUT = 15.0
EXPECTED_SR = 16000
RECORD_SECONDS = 5.0

# Warm-up: descartar audio inicial (solo se graba la “segunda configuración”)
WARMUP_SECONDS = 1.0

# Tamaño de lectura del socket (16 KB) y buffer del kernel (256 KB)
SOCKET_READ_CHUNK = 16384
SOCKET_RCVBUF = 262144


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

    nonce = np.random.bytes(32)
    if not _send_all(conn, nonce):
        print("[auth] Falló el envío del nonce")
        return False

    head = _recv_exact(conn, 6)
    if head is None or head[:5] != b"AUTH1":
        print("[auth] Cabecera AUTH1 inválida o ausente")
        return False

    dev_len = head[5]
    if dev_len < 1 or dev_len > 64:
        print(f"[auth] dev_len inválido: {dev_len}")
        return False

    dev_id = _recv_exact(conn, dev_len)
    tag = _recv_exact(conn, 32)
    if dev_id is None or tag is None:
        print("[auth] Faltan dev_id o tag")
        return False

    mac = hmac.new(SHARED_KEY, nonce + dev_id, hashlib.sha256).digest()
    ok = hmac.compare_digest(mac, tag)
    _send_all(conn, b"OK" if ok else b"NO")
    conn.settimeout(None)
    return ok


def _parse_wav_header(hdr44: bytes):
    if len(hdr44) != 44:
        raise ValueError("Cabecera WAV incompleta")
    if hdr44[0:4] != b"RIFF" or hdr44[8:12] != b"WAVE":
        raise ValueError("No es RIFF/WAVE")
    if hdr44[12:16] != b"fmt ":
        raise ValueError("Falta el subchunk fmt")

    audio_format, = struct.unpack_from("<H", hdr44, 20)
    num_channels, = struct.unpack_from("<H", hdr44, 22)
    sample_rate, = struct.unpack_from("<I", hdr44, 24)
    bits_per_sample, = struct.unpack_from("<H", hdr44, 34)

    if hdr44[36:40] != b"data":
        raise ValueError("Falta el subchunk data")
    if audio_format != 1:
        raise ValueError(f"Formato no PCM: {audio_format}")

    return {"sr": int(sample_rate), "channels": int(num_channels), "bits": int(bits_per_sample)}


def main():
    if len(SHARED_KEY) != 32:
        raise SystemExit(f"[fatal] SHARED_KEY_HEX inválido: {len(SHARED_KEY)} bytes (esperados 32)")

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, SOCKET_RCVBUF)
    except Exception as e:
        print(f"[warn] SO_RCVBUF no disponible: {e}")

    try:
        s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    except Exception:
        pass

    try:
        print(f"[bind] {HOST}:{PORT}")
        s.bind((HOST, PORT))
    except OSError as e:
        print(f"[warn] bind falló: {e}; reintentando en 0.0.0.0")
        s.bind(("0.0.0.0", PORT))

    s.listen(1)
    print("[ok] Esperando conexión del ESP32…")

    conn, addr = s.accept()
    try:
        try:
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        except Exception as e:
            print(f"[net] TCP_NODELAY no se pudo establecer: {e}")
        print(f"[net] Conexión entrante desde {addr}")

        if not _auth_handshake(conn):
            print("[auth] FALLIDO")
            return
        print("[auth] OK")

        # Cabecera WAV
        hdr = _recv_exact(conn, 44)
        if hdr is None:
            print("[wav] No se recibió la cabecera WAV")
            return
        try:
            meta = _parse_wav_header(hdr)
        except Exception as e:
            print(f"[wav] Cabecera inválida: {e}")
            return

        print(f"[wav] sr={meta['sr']} Hz  ch={meta['channels']}  bits={meta['bits']}")
        if meta["channels"] != 1 or meta["bits"] != 16:
            print("[wav] Este servidor solo soporta audio mono de 16-bit")
            return
        if meta["sr"] != EXPECTED_SR:
            print(f"[warn] Frecuencia de muestreo inesperada: {meta['sr']} (se esperaba {EXPECTED_SR})")

        sr = meta["sr"]
        bytes_per_sample = meta["bits"] // 8
        chan = meta["channels"]

        # ---- Warm-up: descarta X segundos antes de grabar ----
        warmup_bytes = int(WARMUP_SECONDS * sr * bytes_per_sample * chan)
        if warmup_bytes > 0:
            print(f"[run] Warm-up: descartando {WARMUP_SECONDS:.2f} s (~{warmup_bytes} bytes)…")
            discarded = 0
            while discarded < warmup_bytes:
                chunk = conn.recv(min(SOCKET_READ_CHUNK, warmup_bytes - discarded))
                if not chunk:
                    print("[net] Conexión cerrada durante warm-up.")
                    return
                discarded += len(chunk)
            print("[run] Warm-up completado. Iniciando grabación…")

        # ---- Grabación real (solo la “segunda configuración”) ----
        target_bytes = int(RECORD_SECONDS * sr * bytes_per_sample * chan)
        print(f"[run] Grabando {RECORD_SECONDS:.1f} s (~{target_bytes} bytes objetivo)…")

        all_audio_data = bytearray()
        last_pct = -1

        while len(all_audio_data) < target_bytes:
            chunk = conn.recv(SOCKET_READ_CHUNK)
            if not chunk:
                print("\n[net] El cliente cerró la conexión durante la grabación.")
                break
            all_audio_data.extend(chunk)

            pct = int((len(all_audio_data) / target_bytes) * 100)
            if pct // 5 != last_pct // 5:
                print(f"\r[run] Progreso: {min(pct, 100)}%", end="")
                last_pct = pct

        print("\n[run] Grabación finalizada.")

        # Alinear a 16-bit mono
        align = bytes_per_sample * chan
        usable = len(all_audio_data) - (len(all_audio_data) % align)
        payload = bytes(all_audio_data[:usable])

        if payload:
            filename = datetime.now().strftime("recording_%Y%m%d_%H%M%S.wav")
            try:
                with wave.open(filename, 'wb') as wf:
                    wf.setnchannels(chan)
                    wf.setsampwidth(bytes_per_sample)
                    wf.setframerate(sr)
                    wf.writeframes(payload)
                print(f"[ok] Muestra guardada en {filename} ({usable} bytes)")
            except Exception as e:
                print(f"[error] No se pudo guardar el archivo WAV: {e}")
        else:
            print("[warn] No se recibieron datos de audio utilizables.")

    except KeyboardInterrupt:
        print("\n[bye] Interrumpido por el usuario.")
    finally:
        try:
            conn.close()
        except Exception:
            pass
        s.close()


if __name__ == "__main__":
    main()