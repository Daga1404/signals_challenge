import os
import wave
import argparse
import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Tuple

def pcm16_to_float32(x_int16: np.ndarray) -> np.ndarray:
    return (x_int16.astype(np.float32) / 32768.0).clip(-1.0, 1.0)

def float32_to_pcm16(x_float: np.ndarray) -> np.ndarray:
    x = np.clip(x_float, -1.0, 1.0)
    return (x * 32767.0).astype("<i2")

def hann_window(N: int) -> np.ndarray:
    n = np.arange(N, dtype=np.float32)
    return 0.5 - 0.5*np.cos(2*np.pi*n/max(N-1,1))

def fft_mag(x: np.ndarray, sr: int) -> Tuple[np.ndarray, np.ndarray]:
    N = x.size
    if N < 2:
        return np.array([0.0]), np.array([0.0])
    xw = x.astype(np.float32) * hann_window(N)
    Nfft = 1 << int(np.ceil(np.log2(N)))
    X = np.fft.rfft(xw, n=Nfft)
    freqs = np.fft.rfftfreq(Nfft, d=1.0/sr)
    mag = np.abs(X).astype(np.float64) / N
    if mag.size > 1:
        mag[1:-1] *= 2.0
    return freqs, mag

def record(seconds: float, sr: int, warmup: float = 0.25) -> np.ndarray:
    sd.default.samplerate = sr
    sd.default.channels = 1
    if warmup > 0:
        _ = sd.rec(int(warmup * sr), dtype="int16")
        sd.wait()
    print(f"[rec] Grabando {seconds:.2f} s a {sr} Hz…")
    rec = sd.rec(int(seconds * sr), dtype="int16")
    sd.wait()
    x = pcm16_to_float32(rec.reshape(-1))
    return x

def save_wav(path: str, x: np.ndarray, sr: int) -> None:
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(sr)
        wf.writeframes((x * 32767.0).astype("<i2").tobytes())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sr", type=int, default=16000, help="Frecuencia de muestreo [Hz]")
    ap.add_argument("--seconds", type=float, default=3.0, help="Duración de la toma [s]")
    ap.add_argument("--outdir", type=str, default=None, help="Carpeta de salida")
    ap.add_argument("--fmax", type=float, default=3500.0, help="Frecuencia máxima para graficar [Hz]")
    args = ap.parse_args()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = args.outdir or f"single_take_{stamp}"
    os.makedirs(outdir, exist_ok=True)
    sr = args.sr
    seconds = args.seconds

    x = record(seconds=seconds, sr=sr, warmup=0.25)

    # recorta/pad exacto
    N = int(seconds * sr)
    if x.size > N: x = x[:N]
    elif x.size < N: x = np.pad(x, (0, N - x.size))

    # Guardar WAV
    wav_path = os.path.join(outdir, "take.wav")
    save_wav(wav_path, x, sr)
    print(f"[ok] WAV guardado en: {wav_path}")

    # Gráfica tiempo
    t = np.arange(x.size) / float(sr)
    fig1, ax1 = plt.subplots(1,1, figsize=(10,4))
    ax1.plot(t, x, linewidth=1.0)
    ax1.grid(True)
    ax1.set_xlabel("Tiempo (s)")
    ax1.set_ylabel("Amplitud")
    ax1.set_title("Señal en el tiempo")
    fig1.tight_layout()
    fig1.savefig(os.path.join(outdir, "time.png"), dpi=150)
    plt.close(fig1)

    # Gráfica frecuencia
    freqs, mag = fft_mag(x, sr)
    sel = freqs <= args.fmax
    fig2, ax2 = plt.subplots(1,1, figsize=(10,4))
    ax2.plot(freqs[sel], mag[sel], linewidth=1.0)
    ax2.grid(True)
    ax2.set_xlim(0, args.fmax)
    ax2.set_xlabel("Frecuencia (Hz)")
    ax2.set_ylabel("Amplitud")
    ax2.set_title("Espectro (FFT)")
    fig2.tight_layout()
    fig2.savefig(os.path.join(outdir, "fft.png"), dpi=150)
    plt.close(fig2)

    print("[done] Listo.")

if __name__ == "__main__":
    main()
