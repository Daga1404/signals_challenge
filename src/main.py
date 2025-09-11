import os
import sys
import glob
import wave
import time
from typing import Optional, Tuple
import numpy as np
from dotenv import load_dotenv

# ============== deps para micrófono ==============
import sounddevice as sd

# ====================== Config ====================
load_dotenv()  # por compatibilidad, aunque ya no usamos HOST/PORT
AUTH_TIMEOUT = 15.0  # sin uso real aquí, se mantiene por compatibilidad

RECORD_TESTS_IF_MISSING = True
N_TESTS = 6
WARMUP_SECONDS = 0.25

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

# === NUEVO: bandas con alta resolución en graves ===
FMIN = 50.0
FSPLIT = 300.0
FMAX_BANDS = 3500.0
N_LOW = 18
N_HIGH = 14

def compute_band_edges(fmin: float = FMIN,
                       fsplit: float = FSPLIT,
                       fmax: float = FMAX_BANDS,
                       n_low: int = N_LOW,
                       n_high: int = N_HIGH) -> np.ndarray:
    low_edges = np.linspace(fmin, fsplit, n_low + 1)          # lineal denso en graves
    high_edges = np.logspace(np.log10(fsplit), np.log10(fmax), n_high + 1)  # log arriba
    edges = np.concatenate([low_edges[:-1], high_edges])       # evita duplicar fsplit
    return edges

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
    import matplotlib.pyplot as plt
    freqs, mag = _fft_mag_linear(x, sr); sel = freqs<=fmax
    fig, ax = plt.subplots(1,1, figsize=(10,4))
    ax.plot(freqs[sel], mag[sel], linewidth=1.0)
    ax.grid(True); ax.set_xlim(0,fmax)
    ax.set_xlabel("Frecuencia (Hz)"); ax.set_ylabel("Amplitud")
    ax.set_title(f"Prueba {j} - FFT (Predicción: Persona {pred_label})")
    fig.tight_layout()
    fname = os.path.join(outdir, f"test_{j}_fft.png")
    fig.savefig(fname, dpi=150); plt.close(fig)
    print(f"[plot] Guardado {fname}")

# ====================== Modelo (centroides) ===================================
def build_model_from_folder(run_dir: str):
    meta_path = os.path.join(run_dir, "meta.npz")
    if not os.path.isfile(meta_path):
        raise SystemExit(f"[fatal] No existe {meta_path}. Genera la carpeta con recolección de muestras")
    meta = np.load(meta_path, allow_pickle=True)
    sr = int(meta["sr"]); take_seconds = float(meta["take_seconds"])
    n_persons = int(meta["n_persons"]); n_takes = int(meta["n_takes_train"])
    names = list(meta["person_names"])
    fmax = float(meta["fmax"])
    print(f"[meta] sr={sr}, personas={n_persons}, tomas/persona={n_takes}")

    band_edges = compute_band_edges()  # NUEVO esquema de bandas

    Nsamples_take = int(sr * take_seconds)
    F_train = []; y_train = []
    for p in range(1, n_persons+1):
        for k in range(1, n_takes+1):
            path = os.path.join(run_dir, f"train_p{p}_t{k}.wav")
            if not os.path.isfile(path):
                raise SystemExit(f"[fatal] Falta {path}")
            x, sr_w = read_wav_float(path)
            assert sr_w == sr, f"{path}: sr distinta ({sr_w})"
            if x.size > Nsamples_take: x = x[:Nsamples_take]
            elif x.size < Nsamples_take: x = np.pad(x, (0, Nsamples_take - x.size))
            # === NUEVO: normalización RMS por toma ===
            rms = float(np.sqrt(np.mean(x**2)) + 1e-12)
            x = (x / rms).clip(-1.0, 1.0)
            fb = features_fft_bands(x, sr, band_edges)
            F_train.append(fb); y_train.append(p-1)

    F_train = np.vstack(F_train).astype(np.float32)
    y_train = np.array(y_train, dtype=np.int32)

    muF = F_train.mean(axis=0, keepdims=True)
    sigmaF = F_train.std(axis=0, keepdims=True) + 1e-9
    Fz = (F_train - muF) / sigmaF
    n_classes = n_persons
    centroids = np.zeros((n_classes, Fz.shape[1]), dtype=np.float32)
    for p in range(n_classes):
        centroids[p, :] = Fz[y_train == p, :].mean(axis=0)

    return {
        "sr": sr, "take_seconds": take_seconds,
        "band_edges": band_edges, "muF": muF, "sigmaF": sigmaF,
        "centroids": centroids, "names": names, "fmax": fmax
    }

def classify_vector(fb: np.ndarray, model: dict):
    fz = (fb - model["muF"].squeeze()) / model["sigmaF"].squeeze()
    dists = np.sqrt(((model["centroids"] - fz) ** 2).sum(axis=1))
    pred = int(np.argmin(dists))
    return pred, dists

# ====================== Grabación por micrófono ================================
def record_one_take_sr(seconds: float, sr: int, warmup_s: float) -> np.ndarray:
    """
    Graba 'seconds' segundos en mono int16 al muestreo 'sr'.
    Hace un breve warmup para estabilizar dispositivos.
    Devuelve float32 en [-1,1] con longitud exacta sr*seconds.
    """
    sd.default.samplerate = sr
    sd.default.channels = 1

    # warmup opcional
    if warmup_s > 0:
        _ = sd.rec(int(warmup_s * sr), dtype="int16")
        sd.wait()

    print(f"[rec] Grabando {seconds:.2f}s a {sr} Hz desde el micrófono…")
    rec = sd.rec(int(seconds * sr), dtype="int16")
    sd.wait()

    x_i16 = rec.reshape(-1)
    x = (x_i16.astype(np.float32) / 32768.0).clip(-1.0, 1.0)
    return x

# ====================== Predicción desde carpeta / Micrófono ===================
def main():
    if len(sys.argv) < 2:
        print("Uso: python prediccion_desde_microfono.py <ruta_run_YYYYMMDD_HHMMSS>")
        sys.exit(1)

    run_dir = sys.argv[1]
    if not os.path.isdir(run_dir):
        print(f("[fatal] Carpeta no encontrada: {run_dir}"))
        sys.exit(1)

    model = build_model_from_folder(run_dir)
    sr = model["sr"]; take_seconds = model["take_seconds"]
    Nsamples_take = int(sr * take_seconds)
    band_edges = model["band_edges"]; names = model["names"]; fmax = model["fmax"]

    # 1) Clasificar si ya hay test_*.wav
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
            # === NUEVO: normalización RMS por toma ===
            rms = float(np.sqrt(np.mean(x**2)) + 1e-12)
            x = (x / rms).clip(-1.0, 1.0)
            fb = features_fft_bands(x, sr, band_edges)
            pred, dists = classify_vector(fb, model)
            print(f"[pred] {os.path.basename(path)} -> {names[pred]}  (distancias: {', '.join(f'{d:.3f}' for d in dists)})")
            plot_test_fft(x, sr, run_dir, i, pred+1, fmax=fmax)
        sys.exit(0)

    # 2) Si no hay test_*.wav y está habilitado, grabamos con micrófono
    if not RECORD_TESTS_IF_MISSING:
        print("[info] No hay test_*.wav y RECORD_TESTS_IF_MISSING=False. Nada que hacer.")
        sys.exit(0)

    print("[mic] No hay pruebas en disco. Usaremos el micrófono local para grabar pruebas.")
    preds = []
    for j in range(1, N_TESTS+1):
        try:
            input(f"\n>> Prueba {j}/{N_TESTS}: Presiona Enter y habla {take_seconds:.1f}s…")
        except EOFError:
            print("[ui] stdin no disponible; continuando sin pausa")

        x = record_one_take_sr(take_seconds, sr, WARMUP_SECONDS)

        if x.size > Nsamples_take: x = x[:Nsamples_take]
        elif x.size < Nsamples_take: x = np.pad(x, (0, Nsamples_take - x.size))

        # === NUEVO: normalización RMS por toma ===
        rms = float(np.sqrt(np.mean(x**2)) + 1e-12)
        x = (x / rms).clip(-1.0, 1.0)

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

if __name__ == "__main__":
    main()
