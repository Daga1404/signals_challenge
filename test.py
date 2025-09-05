import os
import sys
import glob
import wave
import socket
import hmac
import hashlib
import struct
import time
from typing import Optional, Tuple, List
import numpy as np
import matplotlib.pyplot as plt

# ====================== Config comunes (mismo puerto/clave) ====================
HOST = "192.168.137.1"
PORT = 12345
SHARED_KEY_HEX = "83e15c0a2b6a0f6f3a040a9f9b21f3c77e2b6d7d6d6e1e4a2b8c1d2e3f405062"
SHARED_KEY = bytes.fromhex(SHARED_KEY_HEX)
AUTH_TIMEOUT = 15.0

# Si NO hay test_*.wav en la carpeta, podemos grabar aquí:
RECORD_TESTS_IF_MISSING = True
N_TESTS = 4
WARMUP_SECONDS = 0.25
SOCKET_READ_CHUNK = 16384
SOCKET_RCVBUF = 262144

# ====================== Utilidades WAV / features ==============================
def read_wav_float(path: str) -> Tuple[np.ndarray, int]:
    with wave.open(path, "rb") as wf:
        ch = wf.getnchannels(); sw = wf.getsampwidth(); sr = wf.getframerate()
        assert ch == 1 and sw == 2, f"{path}: esperado mono 16-bit"
        frames = wf.readframes(wf.getnframes())
    x = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    return x, sr

def features_fft_bands(x: np.ndarray, sr: int, band_edges: np.ndarray) -> np.ndarray:
    N = x.shape[0]
    if N < 2: return np.zeros(len(band_edges)-1, dtype=np.float32)
    n = np.arange(N, dtype=np.float32)
    w = 0.5 - 0.5*np.cos(2*np.pi*n/max(N-1,1))
    xw = x.astype(np.float32)*w
    Nfft = 1<<int(np.ceil(np.log2(N)))
    X = np.fft.rfft(xw, n=Nfft)
    P = (np.abs(X)**2).astype(np.float64) + 1e-12
    freqs = np.fft.rfftfreq(Nfft, d=1.0/sr)
    fb = np.zeros(len(band_edges)-1, dtype=np.float32)
    for i in range(len(band_edges)-1):
        f1, f2 = band_edges[i], band_edges[i+1]
        idx = (freqs >= f1) & (freqs < f2)
        s = P[idx].sum() if np.any(idx) else 1e-12
        fb[i] = np.log10(s)
    return fb

def compute_band_edges(fmin: float, fmax: float, nbands: int) -> np.ndarray:
    return np.logspace(np.log10(fmin), np.log10(fmax), nbands + 1)

def _fft_mag_linear(x: np.ndarray, sr: int):
    N = x.size
    if N < 2: return np.array([0.0]), np.array([0.0])
    n = np.arange(N, dtype=np.float32)
    w = 0.5 - 0.5*np.cos(2*np.pi*n/max(N-1,1))
    xw = x.astype(np.float32)*w
    Nfft = 1<<int(np.ceil(np.log2(N)))
    X = np.fft.rfft(xw, n=Nfft)
    freqs = np.fft.rfftfreq(Nfft, d=1.0/sr)
    mag = np.abs(X).astype(np.float64)/N
    if mag.size>1: mag[1:-1]*=2.0
    return freqs, mag

def plot_test_fft(x: np.ndarray, sr: int, outdir: str, j: int, pred_label: int, fmax: float):
    freqs, mag = _fft_mag_linear(x, sr); sel = freqs<=fmax
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1,1, figsize=(10,4))
    ax.plot(freqs[sel], mag[sel], linewidth=1.0)
    ax.grid(True); ax.set_xlim(0,fmax)
    ax.set_xlabel("Frecuencia (Hz)"); ax.set_ylabel("Amplitud")
    ax.set_title(f"Prueba {j} - FFT (Predicción: Persona {pred_label})")
    fig.tight_layout()
    fname = os.path.join(outdir, f"test_{j}_fft.png")
    fig.savefig(fname, dpi=150); plt.close(fig)
    print(f"[plot] Guardado {fname}")

# ====================== Red para grabar pruebas (si hace falta) ================
def _recv_exact(conn: socket.socket, n: int) -> Optional[bytes]:
    buf = bytearray()
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk: return None
        buf.extend(chunk)
    return bytes(buf)

def _send_all(conn: socket.socket, data: bytes) -> bool:
    total = 0
    while total < len(data):
        sent = conn.send(data[total:])
        if sent <= 0: return False
        total += sent
    return True

def _auth_handshake(conn: socket.socket, key: bytes, timeout: float = AUTH_TIMEOUT) -> bool:
    conn.settimeout(timeout)
    nonce = np.random.bytes(32)
    if not _send_all(conn, nonce): return False
    head = _recv_exact(conn, 6)
    if head is None or head[:5] != b"AUTH1": return False
    dev_len = head[5]
    if dev_len < 1 or dev_len > 64: return False
    dev_id = _recv_exact(conn, dev_len)
    tag = _recv_exact(conn, 32)
    if dev_id is None or tag is None: return False
    mac = hmac.new(key, nonce + dev_id, hashlib.sha256).digest()
    ok = hmac.compare_digest(mac, tag)
    _send_all(conn, b"OK" if ok else b"NO")
    conn.settimeout(None)
    return ok

def drain_socket(conn: socket.socket, max_drain_sec: float = 0.5) -> int:
    drained = 0
    prev_to = conn.gettimeout()
    try:
        conn.settimeout(0.001); t0 = time.time()
        while time.time()-t0 < max_drain_sec:
            try:
                chunk = conn.recv(SOCKET_READ_CHUNK)
                if not chunk: break
                drained += len(chunk)
            except socket.timeout:
                break
    finally:
        conn.settimeout(prev_to)
    return drained

def recv_audio_bytes(conn: socket.socket, n_bytes: int) -> Optional[bytes]:
    buf = bytearray()
    while len(buf) < n_bytes:
        chunk = conn.recv(min(SOCKET_READ_CHUNK, n_bytes - len(buf)))
        if not chunk: return None
        buf.extend(chunk)
    return bytes(buf)

def float32_to_pcm16(x_float: np.ndarray) -> np.ndarray:
    x = np.clip(x_float, -1.0, 1.0)
    return (x * 32767.0).astype("<i2")

# ====================== Modelo (centroides) ===================================
def build_model_from_folder(run_dir: str):
    # Carga metadatos
    meta_path = os.path.join(run_dir, "meta.npz")
    if not os.path.isfile(meta_path):
        raise SystemExit(f"[fatal] No existe {meta_path}. Genera la carpeta con grabacion_muestras.py")
    meta = np.load(meta_path, allow_pickle=True)
    sr = int(meta["sr"]); take_seconds = float(meta["take_seconds"])
    n_persons = int(meta["n_persons"]); n_takes = int(meta["n_takes_train"])
    names = list(meta["person_names"])
    fmax = float(meta["fmax"])
    print(f"[meta] sr={sr}, personas={n_persons}, tomas/persona={n_takes}")

    # Bandas para features (mismas que en entrenamiento)
    FMIN, FMAX, N_BANDS = 80.0, 5000.0, 12
    band_edges = compute_band_edges(FMIN, FMAX, N_BANDS)

    # Cargar WAVs de entrenamiento
    Nsamples_take = int(sr * take_seconds)
    F_train = []
    y_train = []
    for p in range(1, n_persons+1):
        for k in range(1, n_takes+1):
            path = os.path.join(run_dir, f"train_p{p}_t{k}.wav")
            if not os.path.isfile(path):
                raise SystemExit(f"[fatal] Falta {path}")
            x, sr_w = read_wav_float(path)
            assert sr_w == sr, f"{path}: sr distinta ({sr_w})"
            if x.size > Nsamples_take: x = x[:Nsamples_take]
            elif x.size < Nsamples_take: x = np.pad(x, (0, Nsamples_take - x.size))
            fb = features_fft_bands(x, sr, band_edges)
            F_train.append(fb); y_train.append(p-1)

    F_train = np.vstack(F_train).astype(np.float32)
    y_train = np.array(y_train, dtype=np.int32)

    # z-score + centroides
    muF = F_train.mean(axis=0, keepdims=True)
    sigmaF = F_train.std(axis=0, keepdims=True) + 1e-9
    Fz = (F_train - muF) / sigmaF
    n_classes = n_persons
    centroids = np.zeros((n_classes, Fz.shape[1]), dtype=np.float32)
    for p in range(n_classes):
        centroids[p, :] = Fz[y_train == p, :].mean(axis=0)

    model = {
        "sr": sr, "take_seconds": take_seconds,
        "band_edges": band_edges, "muF": muF, "sigmaF": sigmaF,
        "centroids": centroids, "names": names, "fmax": fmax
    }
    return model

def classify_vector(fb: np.ndarray, model: dict) -> Tuple[int, np.ndarray]:
    fz = (fb - model["muF"].squeeze()) / model["sigmaF"].squeeze()
    dists = np.sqrt(((model["centroids"] - fz) ** 2).sum(axis=1))
    pred = int(np.argmin(dists))  # 0..N-1
    return pred, dists

# ====================== Predicción desde carpeta ========================================
def main():
    if len(sys.argv) < 2:
        print("Uso: python prediccion_desde_carpeta.py <ruta_run_YYYYMMDD_HHMMSS>")
        sys.exit(1)

    run_dir = sys.argv[1]
    if not os.path.isdir(run_dir):
        print(f"[fatal] Carpeta no encontrada: {run_dir}")
        sys.exit(1)

    model = build_model_from_folder(run_dir)
    sr = model["sr"]; take_seconds = model["take_seconds"]
    Nsamples_take = int(sr * take_seconds)
    band_edges = model["band_edges"]; names = model["names"]; fmax = model["fmax"]

    # 1) Si hay test_*.wav en la carpeta, los clasificamos
    test_files = sorted(glob.glob(os.path.join(run_dir, "test_*.wav")))
    if test_files:
        print(f"[info] Encontrados {len(test_files)} archivos de prueba en la carpeta.")
        for i, path in enumerate(test_files, 1):
            x, sr_w = read_wav_float(path)
            if sr_w != sr:
                print(f"[warn] {path}: sr={sr_w} distinta; re-muestrea fuera de este script.")
                continue
            if x.size > Nsamples_take: x = x[:Nsamples_take]
            elif x.size < Nsamples_take: x = np.pad(x, (0, Nsamples_take - x.size))
            fb = features_fft_bands(x, sr, band_edges)
            pred, dists = classify_vector(fb, model)
            print(f"[pred] {os.path.basename(path)} -> {names[pred]}  (distancias: {', '.join(f'{d:.3f}' for d in dists)})")
            plot_test_fft(x, sr, run_dir, i, pred+1, fmax=fmax)
        sys.exit(0)

    # 2) Si no hay test_*.wav y está habilitado, grabamos pruebas nuevas con el ESP32
    if not RECORD_TESTS_IF_MISSING:
        print("[info] No hay test_*.wav y RECORD_TESTS_IF_MISSING=False. Nada que hacer.")
        sys.exit(0)

    print("[net] No hay pruebas en disco. Abriendo servidor para grabar pruebas…")

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try: s.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, SOCKET_RCVBUF)
    except Exception: pass
    try:
        print(f"[bind] {HOST}:{PORT}"); s.bind((HOST, PORT))
    except OSError as e:
        print(f"[warn] bind falló: {e}; reintentando en 0.0.0.0"); s.bind(("0.0.0.0", PORT))
    s.listen(1); print("[ok] Esperando conexión del ESP32…")
    conn, addr = s.accept()

    try:
        try: conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        except Exception: pass
        print(f"[net] Conexión desde {addr}")

        if not _auth_handshake(conn, SHARED_KEY, AUTH_TIMEOUT):
            print("[auth] FALLIDO"); return
        print("[auth] OK")

        # WAV header
        hdr = _recv_exact(conn, 44)
        if hdr is None: print("[wav] No se recibió cabecera WAV"); return
        # Validación simple (no re-usamos todo el parser)
        if hdr[0:4] != b"RIFF" or hdr[8:12] != b"WAVE":
            print("[wav] Cabecera no WAVE"); return

        bytes_per_sample = 2; ch = 1
        align = bytes_per_sample * ch
        Nbytes_take = Nsamples_take * align

        preds = []
        for j in range(1, N_TESTS+1):
            input(f"\n>> Prueba {j}/{N_TESTS}: Presiona Enter y habla {take_seconds:.1f}s…")
            drain_socket(conn, max_drain_sec=0.5)
            warm_bytes = int(WARMUP_SECONDS * sr * align)
            if warm_bytes > 0: _ = recv_audio_bytes(conn, warm_bytes)

            print(f"[rec] Capturando {take_seconds:.1f}s (~{Nbytes_take} bytes)")
            payload = recv_audio_bytes(conn, Nbytes_take)
            if payload is None:
                print("[net] Conexión cerrada durante captura de prueba"); break

            usable = len(payload) - (len(payload) % align)
            payload = payload[:usable]
            x_i16 = np.frombuffer(payload, dtype="<i2")
            x = (x_i16.astype(np.float32) / 32768.0).clip(-1.0, 1.0)
            if x.size > Nsamples_take: x = x[:Nsamples_take]
            elif x.size < Nsamples_take: x = np.pad(x, (0, Nsamples_take - x.size))

            # Guardar WAV en la MISMA carpeta run
            wav_name = os.path.join(run_dir, f"test_{j}.wav")
            with wave.open(wav_name, "wb") as wf:
                wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(sr)
                wf.writeframes((x * 32767.0).astype("<i2").tobytes())
            print(f"[ok] Guardado {wav_name}")

            fb = features_fft_bands(x, sr, band_edges)
            pred, dists = classify_vector(fb, model)
            preds.append(pred+1)
            print(f"[pred] Prueba {j} -> {names[pred]}  (distancias: {', '.join(f'{d:.3f}' for d in dists)})")
            plot_test_fft(x, sr, run_dir, j, pred+1, fmax=fmax)

        if preds:
            with open(os.path.join(run_dir, "predictions.csv"), "w", encoding="utf-8") as f:
                f.write("prueba,prediccion\n")
                for i,p in enumerate(preds,1):
                    f.write(f"{i},{p}\n")
            print(f"[ok] Predicciones guardadas en {os.path.join(run_dir,'predictions.csv')}")

    finally:
        try: conn.close()
        except Exception: pass
        s.close()

if __name__ == "__main__":
    main()