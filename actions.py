# actions.py
import time

# Debounce por banda (ms)
DEBOUNCE_MS = 300
_last_fire = {}

# Define bandas y acciones. Ejemplo:
# (low_Hz, high_Hz, nombre, funcion)
def acceso_david(): print("[ACCION] 300 Hz → Hola David")
def acceso_gabo(): print("[ACCION] 440 Hz → Hola Gabo")
def acceso_dante(): print("[ACCION] 600 Hz → Hola Dante")
def acceso_gal(): print("[ACCION] 880 Hz → Hola Gal")

BANDS = [
    (295.0, 305.0,  "band_1", acceso_david),
    (435.0, 445.0,  "band_2", acceso_gabo),
    (595.0, 605.0,  "band_3", acceso_dante),
    (875.0, 885.0, "band_4", acceso_gal),
    # añade las que necesites
]

def _debounced(key, now_ms):
    last = _last_fire.get(key, 0)
    if now_ms - last < DEBOUNCE_MS:
        return False
    _last_fire[key] = now_ms
    return True

def route_frequency(freq_hz: float, confidence_db: float):
    """
    Decide acción por banda. Puedes usar confidence_db si quieres exigir mayor confianza.
    """
    now = int(time.time() * 1000)
    for low, high, name, func in BANDS:
        if low <= freq_hz <= high:
            if _debounced(name, now):
                func()
            return True
    # Si no cae en ninguna banda, no dispara
    return False
