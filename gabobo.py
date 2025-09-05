import math
import socket
import hmac
import hashlib
import struct
from typing import Optional, Tuple
import wave
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import time

# --------------------------- Configuración de red / seguridad ---------------------------

HOST = "192.168.137.1"
PORT = 12345

SHARED_KEY_HEX = "83e15c0a2b6a0f6f3a040a9f9b21f3c77e2b6d7d6d6e1e4a2b8c1d2e3f405062"
SHARED_KEY = bytes.fromhex(SHARED_KEY_HEX)

AUTH_TIMEOUT = 15.0

# --------------------------- Parámetros de audio / experimento --------------------------

EXPECTED_SR = 16000          # Hz (debes alinear el ESP32 a esto)
TAKE_SECONDS = 3.0           # duración por toma (como en MATLAB)
WARMUP_SECONDS = 0.25        # pequeño flush antes de cada toma tras pulsar Enter

N_PERSONS = 4
N_TAKES_TRAIN = 3            # entrenamiento: 3 tomas por persona
N_TESTS = 4                  # 4 tomas de prueba

# Bandas FFT para features (log-espaciadas)
N_BANDS = 12
FMIN = 80.0
FMAX = 4000.0

# Socket
SOCKET_READ_CHUNK = 16384
SOCKET_RCVBUF = 262144

# --------------------------- Utilidades de red ------------------------------------------

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

def drain_socket(conn: socket.socket, max_drain_sec: float = 0.5) -> int:
    """Lee y descarta lo disponible sin bloquear para 'limpiar' antes de una toma."""
    drained = 0
    prev_to = conn.gettimeout()
    try:
        conn.settimeout(0.001)
        t0 = time.time()
        while time.time() - t0 < max_drain_sec:
            try:
                chunk = conn.recv(SOCKET_READ_CHUNK)
                if not chunk:
                    break
                drained += len(chunk)
            except socket.timeout:
                break
    finally:
        conn.settimeout(prev_to)
    if drained > 0:
        print(f"[drain] Descartados {drained} bytes pendientes")
    return drained

def recv_audio_bytes(conn: socket.socket, n_bytes: int) -> Optional[bytes]:
    """Recibe exactamente n_bytes de audio del socket."""
    buf = bytearray()
    while len(buf) < n_bytes:
        chunk = conn.recv(min(SOCKET_READ_CHUNK, n_bytes - len(buf)))
        if not chunk:
            return None
        buf.extend(chunk)
    return bytes(buf)

def wait_enter(msg: str = "Presiona Enter para continuar..."):
    try:
        input(msg)
    except EOFError:
        print("[ui] stdin no disponible; continuando sin pausa")

# --------------------------- Utilidades de audio / features -----------------------------

def pcm16_to_float32(x_int16: np.ndarray) -> np.ndarray:
    return (x_int16.astype(np.float32) / 32768.0).clip(-1.0, 1.0)

def float32_to_pcm16(x_float: np.ndarray) -> np.ndarray:
    x = np.clip(x_float, -1.0, 1.0)
    return (x * 32767.0).astype(np.int16)

def features_fft_bands(x: np.ndarray, sr: int, band_edges: np.ndarray) -> np.ndarray:
    """Ventaneo Hann, FFT (rfft) y energía log10 integrada por bandas [FMIN..FMAX]."""
    N = x.shape[0]
    if N < 2:
        return np.zeros(len(band_edges) - 1, dtype=np.float32)
    # Hann manual (evita dependencias)
    n = np.arange(N, dtype=np.float32)
    w = 0.5 - 0.5 * np.cos(2.0 * np.pi * n / max(N - 1, 1))
    xw = x.astype(np.float32) * w

    Nfft = 1 << (int(np.ceil(np.log2(N))))
    X = np.fft.rfft(xw, n=Nfft)
    P = (np.abs(X) ** 2).astype(np.float64) + 1e-12  # potencia
    freqs = np.fft.rfftfreq(Nfft, d=1.0 / sr)

    fb = np.zeros(len(band_edges) - 1, dtype=np.float32)
    for i in range(len(band_edges) - 1):
        f1, f2 = band_edges[i], band_edges[i + 1]
        idx = (freqs >= f1) & (freqs < f2)
        s = P[idx].sum() if np.any(idx) else 1e-12
        fb[i] = np.log10(s)
    return fb

def compute_band_edges(fmin: float, fmax: float, nbands: int) -> np.ndarray:
    return np.logspace(np.log10(fmin), np.log10(fmax), nbands + 1)

def plot_person_takes(S: np.ndarray, sr: int, person_idx: int, outdir: str):
    """S: [Nsamples, Ntakes, Npersons] (float32 en [-1,1])"""
    Ns, Nt, _ = S.shape
    t = np.arange(Ns) / float(sr)
    fig, axes = plt.subplots(Nt, 1, figsize=(10, 6), sharex=True)
    if Nt == 1:
        axes = [axes]
    for k in range(Nt):
        axes[k].plot(t, S[:, k, person_idx], linewidth=1.0)
        axes[k].grid(True)
        axes[k].set_ylabel("Amp")
        axes[k].set_title(f"Persona {person_idx+1} - Toma {k+1}")
    axes[-1].set_xlabel("Tiempo (s)")
    fig.suptitle(f"Persona {person_idx+1}: 3 tomas")
    fig.tight_layout()
    fname = os.path.join(outdir, f"persona_{person_idx+1}_tomas.png")
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"[plot] Guardado {fname}")

# --------------------------- Pipeline principal -----------------------------------------

def main():
    if len(SHARED_KEY) != 32:
        raise SystemExit(f"[fatal] SHARED_KEY_HEX inválido: {len(SHARED_KEY)} bytes (esperados 32)")

    # Carpeta de salida
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = f"run_{stamp}"
    os.makedirs(outdir, exist_ok=True)

    # Servidor
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

        # --- Auth
        if not _auth_handshake(conn):
            print("[auth] FALLIDO")
            return
        print("[auth] OK")

        # --- WAV header
        hdr = _recv_exact(conn, 44)
        if hdr is None:
            print("[wav] No se recibió cabecera WAV")
            return
        meta = _parse_wav_header(hdr)
        sr = meta["sr"]
        ch = meta["channels"]
        bits = meta["bits"]
        print(f"[wav] sr={sr} Hz  ch={ch}  bits={bits}")
        if ch != 1 or bits != 16:
            print("[wav] Solo se soporta 16-bit mono")
            return
        if sr != EXPECTED_SR:
            print(f"[warn] sr inesperada: {sr} (esperada {EXPECTED_SR})")

        bytes_per_sample = bits // 8
        align = bytes_per_sample * ch

        Nsamples_take = int(TAKE_SECONDS * sr)
        Nbytes_take = Nsamples_take * align

        band_edges = compute_band_edges(FMIN, FMAX, N_BANDS)

        # --- Contenedores entrenamiento
        S_train = np.zeros((Nsamples_take, N_TAKES_TRAIN, N_PERSONS), dtype=np.float32)
        F_train = np.zeros((N_PERSONS * N_TAKES_TRAIN, N_BANDS), dtype=np.float32)
        y_train = np.zeros((N_PERSONS * N_TAKES_TRAIN,), dtype=np.int32)

        print("\n=== ENTRENAMIENTO (4 personas × 3 tomas) ===")
        for p in range(N_PERSONS):
            for k in range(N_TAKES_TRAIN):
                wait_enter(f"\n>> Persona {p+1} - Toma {k+1}/{N_TAKES_TRAIN}: Presiona Enter y empieza a hablar {TAKE_SECONDS:.1f}s...")
                # Drenar backlog + warm-up
                drain_socket(conn, max_drain_sec=0.5)
                warm_bytes = int(WARMUP_SECONDS * sr * align)
                if warm_bytes > 0:
                    _ = recv_audio_bytes(conn, warm_bytes)  # descartar

                print(f"[rec] Capturando {TAKE_SECONDS:.1f}s (~{Nbytes_take} bytes)")
                payload = recv_audio_bytes(conn, Nbytes_take)
                if payload is None:
                    print("[net] Conexión cerrada durante captura")
                    return
                # Alinear y convertir a float
                usable = len(payload) - (len(payload) % align)
                payload = payload[:usable]
                x_i16 = np.frombuffer(payload, dtype="<i2")  # PCM16 little-endian
                x = pcm16_to_float32(x_i16)
                if x.size > Nsamples_take:
                    x = x[:Nsamples_take]
                elif x.size < Nsamples_take:
                    x = np.pad(x, (0, Nsamples_take - x.size))

                # Guardar WAV
                wav_name = os.path.join(outdir, f"train_p{p+1}_t{k+1}.wav")
                with wave.open(wav_name, "wb") as wf:
                    wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(sr)
                    wf.writeframes(float32_to_pcm16(x).tobytes())
                print(f"[ok] Guardado {wav_name}")

                S_train[:, k, p] = x

        # --- Graficar entrenamiento (3 tomas por persona)
        print("\n[plot] Generando figuras de entrenamiento…")
        for p in range(N_PERSONS):
            plot_person_takes(S_train, sr, p, outdir)

        # --- Features entrenamiento + centroides
        print("[feat] Extrayendo features (bandas FFT)…")
        idx = 0
        for p in range(N_PERSONS):
            for k in range(N_TAKES_TRAIN):
                fb = features_fft_bands(S_train[:, k, p], sr, band_edges)
                F_train[idx, :] = fb
                y_train[idx] = p  # etiqueta 0..3
                idx += 1

        muF = F_train.mean(axis=0, keepdims=True)
        sigmaF = F_train.std(axis=0, keepdims=True) + 1e-9
        Fz = (F_train - muF) / sigmaF

        centroids = np.zeros((N_PERSONS, N_BANDS), dtype=np.float32)
        for p in range(N_PERSONS):
            centroids[p, :] = Fz[y_train == p, :].mean(axis=0)

        # --- PREDICCIÓN
        print("\n=== PREDICCIÓN (4 tomas desconocidas) ===")
        preds = []
        for j in range(N_TESTS):
            wait_enter(f"\n>> Prueba {j+1}/{N_TESTS}: Presiona Enter y habla {TAKE_SECONDS:.1f}s…")
            # Drenar backlog + warm-up
            drain_socket(conn, max_drain_sec=0.5)
            warm_bytes = int(WARMUP_SECONDS * sr * align)
            if warm_bytes > 0:
                _ = recv_audio_bytes(conn, warm_bytes)

            print(f"[rec] Capturando {TAKE_SECONDS:.1f}s (~{Nbytes_take} bytes)")
            payload = recv_audio_bytes(conn, Nbytes_take)
            if payload is None:
                print("[net] Conexión cerrada durante captura de prueba")
                return

            usable = len(payload) - (len(payload) % align)
            payload = payload[:usable]
            x_i16 = np.frombuffer(payload, dtype="<i2")
            x = pcm16_to_float32(x_i16)
            if x.size > Nsamples_take:
                x = x[:Nsamples_take]
            elif x.size < Nsamples_take:
                x = np.pad(x, (0, Nsamples_take - x.size))

            # Guardar WAV de prueba
            wav_name = os.path.join(outdir, f"test_{j+1}.wav")
            with wave.open(wav_name, "wb") as wf:
                wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(sr)
                wf.writeframes(float32_to_pcm16(x).tobytes())
            print(f"[ok] Guardado {wav_name}")

            # Predicción
            fb = features_fft_bands(x, sr, band_edges)
            fz = (fb - muF.squeeze()) / sigmaF.squeeze()
            dists = np.sqrt(((centroids - fz) ** 2).sum(axis=1))
            pred = int(np.argmin(dists))  # 0..3
            preds.append(pred + 1)        # 1..4 para imprimir

            print(f"[pred] Toma de prueba {j+1} -> Persona {pred+1}  (distancias: {', '.join(f'{d:.3f}' for d in dists)})")

        # --- Tabla simple de resultados
        print("\n=== RESULTADOS DE PREDICCIÓN ===")
        for j, pclass in enumerate(preds, 1):
            print(f"Prueba {j}: Persona {pclass}")

        # --- Guardar resumen npz
        np.savez(
            os.path.join(outdir, "summary.npz"),
            sr=sr,
            S_train=S_train, F_train=F_train, y_train=y_train,
            muF=muF, sigmaF=sigmaF, centroids=centroids,
            preds=np.array(preds, dtype=np.int32),
            band_edges=band_edges
        )
        print(f"\n[ok] Artefactos guardados en carpeta: {outdir}")

    except KeyboardInterrupt:
        print("\n[bye] Interrumpido por el usuario.")
    finally:
        try:
            conn.close()
        except Exception:
            pass
        s.close()

# ----------------------------------------------------------------------------------------

if __name__ == "__main__":
    main()
