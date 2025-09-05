"""
actions.py - Frequency routing and action dispatch module

This module defines how detected frequencies are processed and what actions
are triggered based on the frequency detection results from main.py.

You can customize the route_frequency() function to implement your own
frequency-based triggers and actions.
"""

import time
from typing import Dict, List, Tuple

# Global state for debouncing and frequency tracking
_last_action_time = 0.0
_frequency_history: List[Tuple[float, float]] = []  # (timestamp, frequency)
_debounce_interval = 0.5  # seconds between actions

def route_frequency(freq_hz: float, confidence_db: float) -> bool:
    """
    Route detected frequencies to specific actions based on frequency ranges.
    
    This function is called from main.py for every detected frequency.
    Implement your custom logic here to trigger actions based on:
    - Frequency value (freq_hz)
    - Detection confidence (confidence_db)
    - Historical frequency data
    - Time-based conditions
    
    Args:
        freq_hz (float): Detected frequency in Hz
        confidence_db (float): Detection confidence in dB
        
    Returns:
        bool: True if an action was triggered, False otherwise
    """
    global _last_action_time, _frequency_history
    
    current_time = time.time()
    
    # Add to frequency history (keep last 10 seconds)
    _frequency_history.append((current_time, freq_hz))
    _frequency_history = [(t, f) for t, f in _frequency_history if current_time - t <= 10.0]
    
    # Only process if confidence is high enough
    if confidence_db < 5.0:
        return False
    
    # Debouncing: prevent too frequent actions
    if current_time - _last_action_time < _debounce_interval:
        return False
    
    # Example frequency-based actions
    action_triggered = False
    
    # Musical note detection (A4 = 440 Hz and harmonics)
    if _is_musical_note(freq_hz, confidence_db):
        print(f"🎵 Musical note detected: {freq_hz:.1f} Hz ({_freq_to_note(freq_hz)})")
        action_triggered = True
    
    # Voice frequency range detection
    elif _is_voice_frequency(freq_hz, confidence_db):
        print(f"🗣️  Voice frequency detected: {freq_hz:.1f} Hz")
        action_triggered = True
    
    # High-frequency signal detection
    elif _is_high_frequency_signal(freq_hz, confidence_db):
        print(f"📡 High-frequency signal: {freq_hz:.1f} Hz")
        action_triggered = True
    
    # Sustained tone detection
    elif _is_sustained_tone(freq_hz, confidence_db):
        print(f"🔔 Sustained tone detected: {freq_hz:.1f} Hz")
        action_triggered = True
    
    if action_triggered:
        _last_action_time = current_time
    
    return action_triggered


def _is_musical_note(freq_hz: float, confidence_db: float) -> bool:
    """Check if frequency corresponds to a musical note."""
    # Musical notes around A4 (440 Hz) with some tolerance
    musical_freqs = [
        220.0,  # A3
        246.9,  # B3
        261.6,  # C4
        293.7,  # D4
        329.6,  # E4
        349.2,  # F4
        392.0,  # G4
        440.0,  # A4
        493.9,  # B4
        523.3,  # C5
        587.3,  # D5
        659.3,  # E5
        698.5,  # F5
        784.0,  # G5
        880.0,  # A5
    ]
    
    tolerance = 10.0  # Hz tolerance for musical notes
    min_confidence = 8.0  # dB
    
    if confidence_db < min_confidence:
        return False
    
    for note_freq in musical_freqs:
        if abs(freq_hz - note_freq) <= tolerance:
            return True
    
    return False


def _is_voice_frequency(freq_hz: float, confidence_db: float) -> bool:
    """Check if frequency is in typical human voice range."""
    # Human voice fundamental frequency ranges
    # Male: ~85-180 Hz, Female: ~165-265 Hz
    voice_min = 80.0
    voice_max = 300.0
    min_confidence = 6.0
    
    return (voice_min <= freq_hz <= voice_max and 
            confidence_db >= min_confidence)


def _is_high_frequency_signal(freq_hz: float, confidence_db: float) -> bool:
    """Check if frequency is a high-frequency signal."""
    high_freq_min = 2000.0  # 2 kHz and above
    high_freq_max = 8000.0  # Up to 8 kHz (Nyquist limit at 16 kHz)
    min_confidence = 10.0   # Higher confidence required for high frequencies
    
    return (high_freq_min <= freq_hz <= high_freq_max and 
            confidence_db >= min_confidence)


def _is_sustained_tone(freq_hz: float, confidence_db: float) -> bool:
    """Check if current frequency represents a sustained tone."""
    global _frequency_history
    
    if confidence_db < 8.0:
        return False
    
    # Check if similar frequency has been detected recently
    current_time = time.time()
    recent_freqs = [f for t, f in _frequency_history 
                   if current_time - t <= 2.0]  # Last 2 seconds
    
    if len(recent_freqs) < 5:  # Need at least 5 recent detections
        return False
    
    # Check if frequencies are stable (within 5% tolerance)
    tolerance = freq_hz * 0.05
    stable_count = sum(1 for f in recent_freqs 
                      if abs(f - freq_hz) <= tolerance)
    
    # Consider it sustained if 80% of recent detections are similar
    return stable_count >= len(recent_freqs) * 0.8


def _freq_to_note(freq_hz: float) -> str:
    """Convert frequency to musical note name (approximate)."""
    import math
    
    # A4 = 440 Hz reference
    A4 = 440.0
    
    # Calculate semitones from A4
    if freq_hz <= 0:
        return "N/A"
    
    # Calculate semitones using logarithm
    semitones = round(12 * math.log2(freq_hz / A4))
    
    # Note names
    notes = ["A", "A#", "B", "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#"]
    
    # Calculate note and octave
    note_idx = semitones % 12
    octave = 4 + (semitones // 12)
    
    return f"{notes[note_idx]}{octave}"


def get_frequency_stats() -> Dict:
    """Get statistics about detected frequencies."""
    global _frequency_history
    
    if not _frequency_history:
        return {"count": 0, "avg_freq": 0, "freq_range": (0, 0)}
    
    freqs = [f for _, f in _frequency_history]
    
    return {
        "count": len(freqs),
        "avg_freq": sum(freqs) / len(freqs),
        "freq_range": (min(freqs), max(freqs)),
        "recent_freqs": freqs[-5:]  # Last 5 frequencies
    }


def clear_frequency_history():
    """Clear the frequency history buffer."""
    global _frequency_history
    _frequency_history.clear()


# Example custom action functions you can extend:

def trigger_led_action(freq_hz: float):
    """Example: Trigger LED based on frequency."""
    # This is where you could control external hardware
    # e.g., GPIO pins, serial communication, etc.
    print(f"💡 LED action triggered for {freq_hz:.1f} Hz")


def log_frequency_event(freq_hz: float, confidence_db: float):
    """Example: Log frequency events to file."""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"{timestamp}: {freq_hz:.2f} Hz (conf: {confidence_db:.1f} dB)\n"
    
    # You could write to a log file here
    print(f"📝 Logged: {log_entry.strip()}")


def send_network_notification(freq_hz: float):
    """Example: Send network notification."""
    # This is where you could send HTTP requests, MQTT messages, etc.
    print(f"📡 Network notification sent for {freq_hz:.1f} Hz")