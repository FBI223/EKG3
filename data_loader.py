import numpy as np



def is_valid_segment(segment, amplitude_threshold=0.15, snr_threshold=3):
    """Sprawdza, czy segment spełnia kryteria amplitudy i stosunku sygnału do szumu (SNR)."""
    amplitude = np.max(segment) - np.min(segment)
    signal_power = np.mean(segment ** 2)
    noise_power = np.mean((segment - np.mean(segment)) ** 2)
    snr = signal_power / (noise_power + 1e-6)  # Zapobiegnięcie dzieleniu przez zero
    return amplitude > amplitude_threshold and snr > snr_threshold


