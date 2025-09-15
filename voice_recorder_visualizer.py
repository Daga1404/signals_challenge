import pyaudio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import threading
import queue
import sys
from scipy.fft import fft, fftfreq

class AudioRecorderVisualizer:
    def __init__(self):
        # Configuración de audio
        self.CHUNK = 1024  # Tamaño del buffer
        self.FORMAT = pyaudio.paFloat32
        self.CHANNELS = 1
        self.RATE = 44100

        # PyAudio
        self.p = pyaudio.PyAudio()
        self.stream = None
        
        # Queue para datos de audio
        self.audio_queue = queue.Queue()
        
        # Control de grabación
        self.recording = False
        
        # Datos para visualización
        self.audio_data = np.zeros(self.CHUNK)
        self.fft_data = np.zeros(self.CHUNK // 2)
        
        # Setup de matplotlib
        self.setup_plots()
        
    def setup_plots(self):
        """Configurar las gráficas"""
        plt.style.use('dark_background')
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Configurar gráfica de onda de audio
        self.ax1.set_title('Onda de Audio en Tiempo Real', color='white', fontsize=14)
        self.ax1.set_xlabel('Muestras')
        self.ax1.set_ylabel('Amplitud')
        self.ax1.set_xlim(0, self.CHUNK)
        self.ax1.set_ylim(-1, 1)
        self.ax1.grid(True, alpha=0.3)
        
        self.line1, = self.ax1.plot(np.zeros(self.CHUNK), color='cyan', linewidth=1.5)
        
        # Configurar gráfica de FFT
        self.ax2.set_title('Transformada de Fourier (Espectro de Frecuencias)', color='white', fontsize=14)
        self.ax2.set_xlabel('Frecuencia (Hz)')
        self.ax2.set_ylabel('Magnitud')
        
        # Frecuencias para el eje x del FFT
        self.freqs = fftfreq(self.CHUNK, 1/self.RATE)[:self.CHUNK//2]
        self.ax2.set_xlim(0, self.RATE//2)
        self.ax2.set_ylim(0, 1)
        self.ax2.grid(True, alpha=0.3)
        
        self.line2, = self.ax2.plot(self.freqs, np.zeros(self.CHUNK//2), color='orange', linewidth=1.5)
        
        plt.tight_layout()
        
    def audio_callback(self):
        """Callback para capturar audio"""
        while self.recording:
            try:
                data = self.stream.read(self.CHUNK, exception_on_overflow=False)
                audio_array = np.frombuffer(data, dtype=np.float32)
                self.audio_queue.put(audio_array)
            except Exception as e:
                print(f"Error en captura de audio: {e}")
                break
                
    def update_plot(self, frame):
        """Actualizar las gráficas"""
        if not self.audio_queue.empty():
            # Obtener nuevos datos de audio
            self.audio_data = self.audio_queue.get()
            
            # Actualizar gráfica de onda
            self.line1.set_ydata(self.audio_data)
            
            # Calcular FFT
            fft_result = fft(self.audio_data)
            fft_magnitude = np.abs(fft_result[:self.CHUNK//2])
            
            # Normalizar FFT
            if np.max(fft_magnitude) > 0:
                fft_magnitude = fft_magnitude / np.max(fft_magnitude)
            
            # Actualizar gráfica de FFT
            self.line2.set_ydata(fft_magnitude)
            
        return self.line1, self.line2
        
    def start_recording(self):
        """Iniciar grabación y visualización"""
        try:
            # Abrir stream de audio
            self.stream = self.p.open(
                format=self.FORMAT,
                channels=self.CHANNELS,
                rate=self.RATE,
                input=True,
                frames_per_buffer=self.CHUNK
            )
            
            self.recording = True
            
            # Iniciar thread de captura de audio
            self.audio_thread = threading.Thread(target=self.audio_callback)
            self.audio_thread.daemon = True
            self.audio_thread.start()
            
            print("Grabación iniciada. Presiona cualquier tecla para detener...")
            print("Habla cerca del micrófono para ver las visualizaciones")
            
            # Configurar animación
            self.ani = FuncAnimation(
                self.fig, 
                self.update_plot, 
                interval=50,  # Actualizar cada 50ms
                blit=True,
                cache_frame_data=False
            )
            
            # Mostrar las gráficas
            plt.show()
            
        except Exception as e:
            print(f"Error al iniciar grabación: {e}")
            self.stop_recording()
            
    def stop_recording(self):
        """Detener grabación"""
        self.recording = False
        
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            
        self.p.terminate()
        print("Grabación detenida.")
        
    def on_key_press(self, event):
        """Manejar presión de teclas"""
        print(f"Tecla presionada: {event.key}")
        self.stop_recording()
        plt.close('all')
        sys.exit(0)

def main():
    """Función principal"""
    print("Grabador de Voz con Visualización en Tiempo Real")
    print("=" * 50)
    
    try:
        # Crear instancia del grabador
        recorder = AudioRecorderVisualizer()
        
        # Conectar evento de teclado
        recorder.fig.canvas.mpl_connect('key_press_event', recorder.on_key_press)
        
        # Iniciar grabación
        recorder.start_recording()
        
    except KeyboardInterrupt:
        print("\n Programa interrumpido por el usuario.")
    except Exception as e:
        print(f" Error: {e}")
        print("\n Asegúrate de tener instaladas las dependencias:")
        print("   pip install pyaudio numpy matplotlib scipy")

if __name__ == "__main__":
    main()