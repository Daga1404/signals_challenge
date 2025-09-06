import socket
import hmac
import hashlib
import struct
from typing import Optional, List, Tuple
import wave
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import threading
from dotenv import load_dotenv

# ====================== Configuración de red / seguridad ======================
load_dotenv()  # busca el archivo .env automáticamente

HOST = os.getenv("ESP_HOST")
PORT = int(os.getenv("ESP_PORT"))
SHARED_KEY_HEX = str(os.getenv("ESP_SHARED_KEY_HEX"))

SHARED_KEY = bytes.fromhex(SHARED_KEY_HEX)
AUTH_TIMEOUT = 15.0

# ====================== Parámetros del experimento ============================
EXPECTED_SR = 16000
TAKE_SECONDS = 3.0
WARMUP_SECONDS = 0.25

N_PERSONS = 3
N_TAKES_TRAIN = 5
PERSON_NAMES: List[str] = ["david", "gal", "gabo"]

# FFT para gráficas (lineal)
FMAX = 5000.0

# Socket
SOCKET_READ_CHUNK = 16384
SOCKET_RCVBUF = 262144

# ====================== Utilidades de red =====================================
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
        print("[auth] Falló envío de nonce")
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

# ====================== Búfer circular con lector continuo ====================
class SocketRingReader:
    def __init__(self, conn: socket.socket, chunk: int, max_bytes: int):
        self.conn = conn
        self.chunk = chunk
        self.max_bytes = max_bytes
        self.buf = bytearray()
        self.lock = threading.Lock()
        self.cv = threading.Condition(self.lock)
        self.running = False
        self.closed = False
        self.thread = None

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def stop(self):
        with self.lock:
            self.running = False
            self.cv.notify_all()
        try:
            if self.thread:
                self.thread.join(timeout=1.0)
        except Exception:
            pass

    def _run(self):
        while True:
            if not self.running:
                break
            try:
                data = self.conn.recv(self.chunk)
            except Exception:
                data = b""
            if not data:
                with self.lock:
                    self.closed = True
                    self.cv.notify_all()
                break
            with self.lock:
                self.buf.extend(data)
                # recorta lo más antiguo si excede el máximo
                if len(self.buf) > self.max_bytes:
                    drop = len(self.buf) - self.max_bytes
                    del self.buf[:drop]
                self.cv.notify_all()

    def clear(self):
        with self.lock:
            self.buf.clear()

    def drop(self, n: int):
        with self.lock:
            n = min(n, len(self.buf))
            del self.buf[:n]

    def read_exact(self, n: int, timeout: float = None) -> Optional[bytes]:
        """Bloquea hasta obtener n bytes o EOF. timeout opcional en segundos."""
        end = None if timeout is None else (time.monotonic() + timeout)
        with self.lock:
            while len(self.buf) < n and not self.closed:
                remaining = None if end is None else max(0.0, end - time.monotonic())
                if end is not None and remaining == 0.0:
                    break
                self.cv.wait(timeout=remaining)
            if len(self.buf) < n and self.closed:
                # devuelve lo que haya si algo quedó, si no None
                if len(self.buf) == 0:
                    return None
                n = len(self.buf)
            out = bytes(self.buf[:n])
            del self.buf[:n]
            return out

# ====================== Utilidades de audio / gráficas =========================
def pcm16_to_float32(x_int16: np.ndarray) -> np.ndarray:
    return (x_int16.astype(np.float32) / 32768.0).clip(-1.0, 1.0)

def float32_to_pcm16(x_float: np.ndarray) -> np.ndarray:
    x = np.clip(x_float, -1.0, 1.0)
    return (x * 32767.0).astype(np.int16)

def _fft_mag_linear(x: np.ndarray, sr: int) -> Tuple[np.ndarray, np.ndarray]:
    N = x.size
    if N < 2:
        return np.array([0.0]), np.array([0.0])
    n = np.arange(N, dtype=np.float32)
    w = 0.5 - 0.5 * np.cos(2.0 * np.pi * n / max(N - 1, 1))
    xw = x.astype(np.float32) * w
    Nfft = 1 << int(np.ceil(np.log2(N)))
    X = np.fft.rfft(xw, n=Nfft)
    freqs = np.fft.rfftfreq(Nfft, d=1.0 / sr)
    mag = np.abs(X).astype(np.float64) / N
    if mag.size > 1:
        mag[1:-1] *= 2.0
    return freqs, mag

def plot_person_takes(S: np.ndarray, sr: int, person_idx: int, outdir: str, names: List[str]):
    Ns, Nt, _ = S.shape
    t = np.arange(Ns) / float(sr)
    fig_h = max(6, int(Nt * 1.6))
    fig, axes = plt.subplots(Nt, 1, figsize=(10, fig_h), sharex=True)
    if Nt == 1: axes = [axes]
    for k in range(Nt):
        axes[k].plot(t, S[:, k, person_idx], linewidth=1.0)
        axes[k].grid(True)
        axes[k].set_ylabel("Amplitud")
        axes[k].set_title(f"{names[person_idx]} - Toma {k+1}")
    axes[-1].set_xlabel("Tiempo (s)")
    fig.suptitle(f"{names[person_idx]}: {Nt} tomas (Tiempo)")
    fig.tight_layout()
    fname = os.path.join(outdir, f"persona_{person_idx+1}_tomas_time.png")
    fig.savefig(fname, dpi=150); plt.close(fig)
    print(f"[plot] Guardado {fname}")

def plot_person_fft(S: np.ndarray, sr: int, person_idx: int, outdir: str, names: List[str], fmax: float = FMAX):
    _, Nt, _ = S.shape
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    for k in range(Nt):
        freqs, mag = _fft_mag_linear(S[:, k, person_idx], sr)
        sel = freqs <= fmax
        ax.plot(freqs[sel], mag[sel], linewidth=1.0, label=f"Toma {k+1}")
    ax.grid(True); ax.set_xlim(0, fmax)
    ax.set_xlabel("Frecuencia (Hz)"); ax.set_ylabel("Amplitud")
    ax.set_title(f"Comparación del Espectro de Frecuencia para {names[person_idx]}")
    ax.legend(loc="upper right", fontsize=8 if Nt > 6 else 10)
    fig.tight_layout()
    try: fig.canvas.manager.set_window_title(f"Espectro de Frecuencia (FFT) - {names[person_idx]}")
    except Exception: pass
    fname = os.path.join(outdir, f"{names[person_idx].replace(' ','_').lower()}_fft.png")
    fig.savefig(fname, dpi=150); plt.close(fig)
    print(f"[plot] Guardado {fname}")

# ====================== Pipeline: GRABACIÓN ====================================
def main():
    if len(SHARED_KEY) != 32:
        raise SystemExit(f"[fatal] SHARED_KEY_HEX inválido: {len(SHARED_KEY)} bytes (esperados 32)")

    # Carpeta de salida
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = f"run_{stamp}"
    os.makedirs(outdir, exist_ok=True)
    print(f"[out] Carpeta: {outdir}")

    # Servidor
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try: s.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, SOCKET_RCVBUF)
    except Exception as e: print(f"[warn] SO_RCVBUF no disponible: {e}")
    try: s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    except Exception: pass

    try:
        print(f"[bind] {HOST}:{PORT}"); s.bind((HOST, PORT))
    except OSError as e:
        print(f"[warn] bind falló: {e}; reintentando en 0.0.0.0"); s.bind(("0.0.0.0", PORT))

    s.listen(1); print("[ok] Esperando conexión del ESP32…")
    conn, addr = s.accept()

    try:
        try: conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        except Exception as e: print(f"[net] TCP_NODELAY no se pudo establecer: {e}")
        print(f"[net] Conexión desde {addr}")

        if not _auth_handshake(conn):
            print("[auth] FALLIDO"); return
        print("[auth] OK")

        # WAV header
        hdr = _recv_exact(conn, 44)
        if hdr is None: print("[wav] No se recibió cabecera WAV"); return
        meta = _parse_wav_header(hdr)
        sr, ch, bits = meta["sr"], meta["channels"], meta["bits"]
        print(f"[wav] sr={sr} Hz, ch={ch}, bits={bits}")
        if ch != 1 or bits != 16: print("[wav] Solo 16-bit mono"); return
        if sr != EXPECTED_SR: print(f"[warn] sr inesperada: {sr} (esperada {EXPECTED_SR})")

        bytes_per_sample = bits // 8
        align = bytes_per_sample * ch
        Nsamples_take = int(TAKE_SECONDS * sr)
        Nbytes_take = Nsamples_take * align

        # ===== iniciar lector continuo con buffer circular =====
        bytes_per_sec = sr * align
        max_buffer_seconds = 8.0  # capacidad del ring buffer (seg)
        ring = SocketRingReader(conn, chunk=SOCKET_READ_CHUNK,
                                max_bytes=int(max_buffer_seconds * bytes_per_sec))
        ring.start()

        # Contenedor para gráficas
        S_train = np.zeros((Nsamples_take, N_TAKES_TRAIN, N_PERSONS), dtype=np.float32)

        print(f"\n=== GRABACIÓN ( {N_PERSONS} personas × {N_TAKES_TRAIN} tomas ) ===")
        for p in range(N_PERSONS):
            for k in range(N_TAKES_TRAIN):
                nombre = PERSON_NAMES[p] if p < len(PERSON_NAMES) else f"Persona {p+1}"
                try:
                    input(f"\n>> {nombre} - Toma {k+1}/{N_TAKES_TRAIN}: Presiona Enter y habla {TAKE_SECONDS:.1f}s…")
                except EOFError:
                    print("[ui] stdin no disponible; continuando sin pausa")

                # limpiar backlog antiguo y descartar warm-up
                ring.clear()
                warm_bytes = int(WARMUP_SECONDS * sr * align)
                if warm_bytes > 0:
                    _ = ring.read_exact(warm_bytes, timeout=2.0)

                print(f"[rec] Capturando {TAKE_SECONDS:.1f}s (~{Nbytes_take} bytes)")
                payload = ring.read_exact(Nbytes_take, timeout=TAKE_SECONDS + 2.0)
                if payload is None or len(payload) == 0:
                    print("[net] Conexión cerrada o timeout durante captura"); return

                usable = len(payload) - (len(payload) % align)
                payload = payload[:usable]
                x_i16 = np.frombuffer(payload, dtype="<i2")
                x = pcm16_to_float32(x_i16)
                if x.size > Nsamples_take: x = x[:Nsamples_take]
                elif x.size < Nsamples_take: x = np.pad(x, (0, Nsamples_take - x.size))

                # Guardar WAV
                wav_name = os.path.join(outdir, f"train_p{p+1}_t{k+1}.wav")
                with wave.open(wav_name, "wb") as wf:
                    wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(sr)
                    wf.writeframes((x * 32767.0).astype("<i2").tobytes())
                print(f"[ok] Guardado {wav_name}")

                S_train[:, k, p] = x

        # detener lector antes de graficar/cerrar
        ring.stop()

        # Gráficas por persona (tiempo + FFT superpuesta)
        print("\n[plot] Generando figuras…")
        for p in range(N_PERSONS):
            plot_person_takes(S_train, sr, p, outdir, PERSON_NAMES)
            plot_person_fft(S_train, sr, p, outdir, PERSON_NAMES, fmax=FMAX)

        # Guardar metadatos
        np.savez(os.path.join(outdir, "meta.npz"),
                 sr=sr, take_seconds=TAKE_SECONDS, n_persons=N_PERSONS,
                 n_takes_train=N_TAKES_TRAIN, person_names=np.array(PERSON_NAMES, dtype=object),
                 fmax=FMAX)
        print(f"\n[ok] Metadatos guardados en {outdir}/meta.npz")
        print("[done] Grabación completada.")

    finally:
        try:
            conn.shutdown(socket.SHUT_RD)
        except Exception:
            pass
        try:
            conn.close()
        except Exception:
            pass
        s.close()

if __name__ == "__main__":
    main()
