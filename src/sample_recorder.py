import os
import wave
import time
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from typing import List, Tuple
from dotenv import load_dotenv

# ============== deps para micrófono ==============
import sounddevice as sd

# ====================== Parámetros del experimento ============================
EXPECTED_SR    = 16000
TAKE_SECONDS   = 3.0
WARMUP_SECONDS = 0.25

N_PERSONS     = 4
N_TAKES_TRAIN = 12
PERSON_NAMES: List[str] = ["David", "Gabo", "Dante", "Gal"]

# FFT para gráficas (lineal)
FMAX = 3500.0  # AJUSTADO para enfocarnos en banda útil de timbre

# ====================== Utilidades de audio / gráficas ========================
def pcm16_to_float32(x_int16: np.ndarray) -> np.ndarray:
    return (x_int16.astype(np.float32) / 32768.0).clip(-1.0, 1.0)

def float32_to_pcm16(x_float: np.ndarray) -> np.ndarray:
    x = np.clip(x_float, -1.0, 1.0)
    return (x * 32767.0).astype("<i2")

def _fft_mag_linear(x: np.ndarray, sr: int) -> Tuple[np.ndarray, np.ndarray]:
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

# ====================== Grabación por micrófono ================================
def record_one_take_sr(seconds: float, sr: int, warmup_s: float) -> np.ndarray:
    """
    Graba 'seconds' segundos en mono int16 al muestreo 'sr'.
    Hace un breve warmup para estabilizar el dispositivo.
    Devuelve float32 en [-1,1] con longitud exacta sr*seconds.
    """
    sd.default.samplerate = sr
    sd.default.channels = 1

    if warmup_s > 0:
        _ = sd.rec(int(warmup_s * sr), dtype="int16")
        sd.wait()

    print(f"[rec] Grabando {seconds:.2f}s a {sr} Hz desde el micrófono…")
    rec = sd.rec(int(seconds * sr), dtype="int16")
    sd.wait()
    x_i16 = rec.reshape(-1)
    x = pcm16_to_float32(x_i16)
    return x

# ====================== Pipeline: RECOLECCIÓN =================================
def main():
    load_dotenv()  # por compatibilidad si usas .env para otras cosas

    # Carpeta de salida
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = f"run_{stamp}"
    os.makedirs(outdir, exist_ok=True)
    print(f"[out] Carpeta: {outdir}")

    sr = EXPECTED_SR
    Nsamples_take = int(TAKE_SECONDS * sr)

    # Contenedor para gráficas
    S_train = np.zeros((Nsamples_take, N_TAKES_TRAIN, N_PERSONS), dtype=np.float32)

    print(f"\n=== GRABACIÓN ( {N_PERSONS} personas × {N_TAKES_TRAIN} tomas ) ===")
    for p in range(N_PERSONS):
        for k in range(N_TAKES_TRAIN):
            nombre = PERSON_NAMES[p] if p < len(PERSON_NAMES) else f"Persona {p+1}"
            try:
                input(f"\n>> {nombre} - Toma {k+1}/{N_TAKES_TRAIN}: Presiona Enter y habla {TAKE_SECONDS:.1f}s…")
            except EOFError:
                print("[ui] stdin no disponible; continuando")

            x = record_one_take_sr(TAKE_SECONDS, sr, WARMUP_SECONDS)

            # asegurar longitud exacta
            if x.size > Nsamples_take: x = x[:Nsamples_take]
            elif x.size < Nsamples_take: x = np.pad(x, (0, Nsamples_take - x.size))

            # Guardar WAV (mono 16-bit, sin normalizar: guardamos la captura cruda)
            wav_name = os.path.join(outdir, f"train_p{p+1}_t{k+1}.wav")
            with wave.open(wav_name, "wb") as wf:
                wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(sr)
                wf.writeframes(float32_to_pcm16(x).tobytes())
            print(f"[ok] Guardado {wav_name}")

            S_train[:, k, p] = x

    # Gráficas por persona (tiempo + FFT)
    print("\n[plot] Generando figuras…")
    for p in range(N_PERSONS):
        plot_person_takes(S_train, sr, p, outdir, PERSON_NAMES)
        plot_person_fft(S_train, sr, p, outdir, PERSON_NAMES, fmax=FMAX)

    # Guardar metadatos (incluye FMAX ajustado)
    np.savez(os.path.join(outdir, "meta.npz"),
             sr=sr, take_seconds=TAKE_SECONDS, n_persons=N_PERSONS,
             n_takes_train=N_TAKES_TRAIN, person_names=np.array(PERSON_NAMES, dtype=object),
             fmax=FMAX)
    print(f"\n[ok] Metadatos guardados en {outdir}/meta.npz")
    print("[done] Recolección completada.")

if __name__ == "__main__":
    main()
