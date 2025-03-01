import numpy as np


def filter_low_amplitude_segments(segments, labels, amplitude_threshold=0.15):
    """ Odrzuca segmenty o zbyt niskiej amplitudzie. """
    valid_segments, valid_labels = [], []

    for i in range(len(segments)):
        segment = segments[i]
        amplitude = np.max(segment) - np.min(segment)

        if amplitude > amplitude_threshold:
            valid_segments.append(segment)
            valid_labels.append(labels[i])

    return np.array(valid_segments), np.array(valid_labels)

def filter_high_noise_segments(segments, labels, snr_threshold=3):
    """ Odrzuca segmenty o dużym poziomie szumu na podstawie stosunku sygnału do szumu (SNR). """
    valid_segments, valid_labels = [], []

    for i in range(len(segments)):
        segment = segments[i]
        signal_power = np.mean(segment ** 2)
        noise_power = np.mean((segment - np.mean(segment)) ** 2)
        snr = signal_power / (noise_power + 1e-6)  # Zapobiegnięcie dzieleniu przez zero

        if snr > snr_threshold:
            valid_segments.append(segment)
            valid_labels.append(labels[i])

    return np.array(valid_segments), np.array(valid_labels)
