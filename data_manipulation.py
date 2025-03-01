import numpy as np
from imblearn.over_sampling import RandomOverSampler
from constants import SEGMENT_LENGTH


def balance_classes_oversampling(X, y):
    """🔄 Oversampling klas mniejszościowych z dodanym szumem do liczby próbek klasy dominującej."""
    ros = RandomOverSampler(sampling_strategy='auto', random_state=42)
    X_resampled, y_resampled = ros.fit_resample(X.reshape(len(X), -1), y)

    return X_resampled.reshape(len(X_resampled), SEGMENT_LENGTH), y_resampled


def balance_classes(X, y, class_to_reduce=0, reduction_factor=0.5):
    """🔄 Redukcja liczby segmentów klasy `class_to_reduce`."""
    idx_class = np.where(y == class_to_reduce)[0]  # Znajdź indeksy klasy "N"
    num_to_remove = int(len(idx_class) * reduction_factor)  # Określ liczbę do usunięcia

    idx_remove = np.random.choice(idx_class, num_to_remove, replace=False)  # Wylosuj do usunięcia
    idx_keep = np.setdiff1d(np.arange(len(y)), idx_remove)  # Indeksy, które zostawiamy

    return X[idx_keep], y[idx_keep]



def balance_classes_smart(X, y):
    """Umiarkowane balansowanie: redukcja klasy 0 + augmentacja mniejszych klas"""
    X, y = balance_classes(X, y, class_to_reduce=0, reduction_factor=0.75 )  # Redukcja klasy 0



    # Augmentacja rzadkich klas (1 i 3, bo mają niski recall)
    rare_classes = [1]
    for cls in rare_classes:
        idx = np.where(y == cls)[0]
        if len(idx) == 0:
            continue  # Pomijamy jeśli nie ma próbek

        X_aug, y_aug = augment_data(X[idx], y[idx], augmentation_factor=3)  # Powiel 3x
        X = np.concatenate((X, X_aug), axis=0)
        y = np.concatenate((y, y_aug), axis=0)



    # Shuffle po balansowaniu
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    X, y = X[indices], y[indices]

    return X, y


def augment_signal(signal, noise_level=0.01, shift=5, scale_factor=0.05):
    """Dodaje szum, skalowanie amplitudy i przesunięcie fazowe do sygnału EKG."""
    noise = np.random.normal(0, noise_level, signal.shape)
    shift_val = np.random.randint(-shift, shift)
    scale = 1 + np.random.uniform(-scale_factor, scale_factor)

    # Przesunięcie fazowe (cykliczne)
    augmented_signal = np.roll(signal, shift_val)

    # Skalowanie i dodanie szumu
    augmented_signal = scale * augmented_signal + noise
    return augmented_signal



def augment_data(X, y, augmentation_factor=2):
    """Tworzy dodatkowe próbki przez augmentację danych."""
    X_aug, y_aug = [], []

    for i in range(len(X)):
        for _ in range(augmentation_factor):  # Powiel dane augmentation_factor razy
            X_aug.append(augment_signal(X[i]))
            y_aug.append(y[i])

    return np.array(X_aug), np.array(y_aug)
