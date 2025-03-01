import numpy as np
import pywt
from scipy.signal import butter, filtfilt, sosfilt, iirnotch, medfilt


def bandpass_filter(signal, fs, lowcut=0.5, highcut=40, order=4):
    """📌 Filtr pasmowo-przepustowy (0.5–40 Hz) do usunięcia zakłóceń mięśniowych i drgań."""
    nyq = 0.5 * fs
    if lowcut >= highcut or highcut >= nyq:
        raise ValueError("Niepoprawne wartości filtracji pasmowo-przepustowej: lowcut < highcut < Nyquist")

    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], btype='bandpass', output='sos')
    return sosfilt(sos, signal)

def notch_filter(signal, fs, freq=50, quality_factor=30):
    """📌 Filtr Notch do usunięcia zakłóceń sieciowych (50/60 Hz)."""
    nyq = 0.5 * fs
    if freq >= nyq:
        raise ValueError("Częstotliwość Notch musi być mniejsza niż Nyquist")

    w0 = freq / nyq
    b, a = iirnotch(w0, quality_factor)
    return filtfilt(b, a, signal)

def highpass_filter(signal, fs, lowcut=0.67, order=4):
    """📌 Filtr górnoprzepustowy (usuwa dryf bazowy poniżej 0.67 Hz)."""
    nyq = 0.5 * fs
    if lowcut >= nyq:
        raise ValueError("Częstotliwość odcięcia highpass musi być mniejsza niż Nyquist")

    low = lowcut / nyq
    sos = butter(order, low, btype='highpass', output='sos')
    return sosfilt(sos, signal)

def wavelet_denoising(signal, wavelet='bior6.8', level=5):
    """📌 Usuwa szum mięśniowy za pomocą DWT (Dekompozycja falkowa)."""
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    threshold = sigma * np.sqrt(2 * np.log(len(signal)))
    coeffs_thresh = [pywt.threshold(c, threshold, mode='soft') for c in coeffs]
    return pywt.waverec(coeffs_thresh, wavelet)

def filter_ecg(signal, fs):
    """📌 Kompleksowa filtracja sygnału EKG"""
    signal = medfilt(signal, kernel_size=3)  # Redukcja nagłych skoków
    signal = bandpass_filter(signal, fs)
    signal = notch_filter(signal, fs, freq=50)
    signal = highpass_filter(signal, fs)
    signal = wavelet_denoising(signal)
    return signal
