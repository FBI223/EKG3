import os
import wfdb
import matplotlib.pyplot as plt
import tensorflow as tf
from imblearn.over_sampling import RandomOverSampler
from scipy.signal import resample
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import pywt
from scipy.signal import butter, filtfilt, sosfilt, iirnotch
import seaborn as sns
import signal
import sys
import tensorflow.keras.backend as K
import gc
from scipy.signal import medfilt

def cleanup_resources(signum, frame):
    print("🛑 Przerywanie... zwalniam pamięć!")
    K.clear_session()
    gc.collect()
    sys.exit(0)

signal.signal(signal.SIGINT, cleanup_resources)  # Obsługa Ctrl+C

# 📂 Foldery z danymi MITDB i SVDB
MITDB_PATH = "mitdb/"
SVDB_PATH = "svdb/"
INCARTDB_PATH = "incartdb/"

# 🔹 Docelowa częstotliwość próbkowania
TARGET_FS = 360
SEGMENT_LENGTH = 300  # Długość segmentu w próbkach (QRS w środku)

# 🔹 Mapowanie etykiet
LABEL_MAP = {
    'N': 0,  # Normal Beats (N)
    'L': 0,  # Left Bundle Branch Block Beat
    'R': 0,  # Right Bundle Branch Block Beat
    'e': 0,  # Atrial Escape Beat
    'j': 0,  # Nodal (Junctional) Escape Beat

    'A': 1,  # Atrial Premature Beat (SVEB)
    'a': 1,  # Aberrated Atrial Premature Beat (SVEB)
    'J': 1,  # Nodal (Junctional) Premature Beat (SVEB)
    'S': 1,  # Supraventricular Premature Beat (SVEB)

    'V': 2,  # Premature Ventricular Contraction (VEB)
    'E': 2,  # Ventricular Escape Beat (VEB)


    'f': 3,  # Fusion of Paced and Normal Beat (Fusion Class)
    'Q': 3,  # Unclassified Beats (Other)
    '/': 3,  # Paced beat
    '?': 3   # Beat not classified during learning

}

NUM_CLASSES = len(set(LABEL_MAP.values()))


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



def plot_confusion_matrix(y_true, y_pred, labels):
    """Rysuje macierz błędów."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()




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
    X, y = balance_classes(X, y, class_to_reduce=0, reduction_factor=0.5)  # Redukcja klasy 0

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



def select_best_lead(record):
    if record.p_signal is None:
        return None

    leads = record.sig_name  # Lista nazw dostępnych kanałów

    if "MLII" in leads:
        return record.p_signal[:, leads.index("MLII")]
    elif "II" in leads:
        return record.p_signal[:, leads.index("II")]
    elif "ECG1" in leads:
        return record.p_signal[:, leads.index("ECG1")]
    else:
        return None


### 🔥 **2. Resampling sygnału**
def resample_ecg_signal(signal, annotation_samples, original_fs, target_fs=TARGET_FS):
    """🔄 Resampling sygnału do docelowej częstotliwości"""
    new_length = int(len(signal) * (target_fs / original_fs))
    resampled_signal = resample(signal, new_length)
    scale_factor = target_fs / original_fs
    resampled_annotations = np.round(np.array(annotation_samples) * scale_factor).astype(int)
    return resampled_signal, resampled_annotations




### 🔥 **3. Wczytywanie i przetwarzanie danych**
def load_ecg_data(db_path, record_ids):
    signals, labels = [], []

    for record_id in record_ids:
        record = wfdb.rdrecord(f'{db_path}/{record_id}')
        annotation = wfdb.rdann(f'{db_path}/{record_id}', 'atr')
        #signal = record.p_signal[:, 0]  # Pobranie 1. odprowadzenia


        # Wybór najlepszego odprowadzenia
        signal = select_best_lead(record)
        if signal is None:
            continue  # Pominięcie rekordu jeśli nie ma MLII, II ani ECG1

        original_fs = record.fs  # Oryginalna częstotliwość próbkowania

        # 🔹 Resampling do TARGET_FS
        if original_fs != TARGET_FS:
            signal, annotation.sample = resample_ecg_signal(signal, annotation.sample, original_fs, TARGET_FS)

        # 🔹 Filtracja sygnału
        signal = filter_ecg(signal, TARGET_FS)

        # 🔹 Segmentacja QRS w środku
        for i, r in enumerate(annotation.sample):
            label = annotation.symbol[i]
            if label in LABEL_MAP:
                start = max(0, r - SEGMENT_LENGTH // 2)
                end = min(len(signal), r + SEGMENT_LENGTH // 2)

                segment = signal[start:end]
                segment_len = len(segment)

                if segment_len < SEGMENT_LENGTH:
                    pad_left = (SEGMENT_LENGTH - segment_len) // 2
                    pad_right = SEGMENT_LENGTH - segment_len - pad_left
                    segment = np.pad(segment, (pad_left, pad_right), mode='edge')

                if len(segment) == SEGMENT_LENGTH:
                    signals.append(segment)
                    labels.append(LABEL_MAP[label])

    return np.array(signals), np.array(labels)



def load_all_ecg_datav1(mitdb_path, svdb_path):
    """🔹 Wczytuje i łączy dane EKG z MITDB oraz SVDB, pomijając INCARTDB."""

    svdb_records = sorted([f.split('.')[0] for f in os.listdir(svdb_path) if f.endswith('.hea')])
    svdb_signals, svdb_labels = load_ecg_data(svdb_path, svdb_records)

    mitdb_records = sorted([f.split('.')[0] for f in os.listdir(mitdb_path) if f.endswith('.hea')])
    mitdb_signals, mitdb_labels = load_ecg_data(mitdb_path, mitdb_records)

    # 🔹 Połączenie zbiorów MITDB i SVDB
    X = np.concatenate((mitdb_signals, svdb_signals), axis=0)
    y = np.concatenate((mitdb_labels, svdb_labels), axis=0)

    # 🔹 Mieszanie danych, zachowując przypisanie etykiet
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    X, y = X[indices], y[indices]

    # 🔹 Normalizacja całego połączonego zbioru danych
    X = (X - np.mean(X)) / np.std(X)

    return X, y


def load_all_ecg_data(mitdb_path, svdb_path, incartdb_path):

    svdb_records = sorted([f.split('.')[0] for f in os.listdir(svdb_path) if f.endswith('.hea')])
    svdb_signals, svdb_labels = load_ecg_data(svdb_path, svdb_records)

    mitdb_records = sorted([f.split('.')[0] for f in os.listdir(mitdb_path) if f.endswith('.hea')])
    mitdb_signals, mitdb_labels = load_ecg_data(mitdb_path, mitdb_records)


    incartdb_records = sorted([f.split('.')[0] for f in os.listdir(incartdb_path) if f.endswith('.hea')])
    incartdb_signals, incartdb_labels = load_ecg_data(incartdb_path, incartdb_records)

    # 🔹 Połączenie wszystkich zbiorów
    X = np.concatenate((mitdb_signals, svdb_signals, incartdb_signals), axis=0)
    y = np.concatenate((mitdb_labels, svdb_labels, incartdb_labels), axis=0)

    #X = mitdb_signals
    #y = mitdb_labels


    # 🔹 Mieszanie danych zachowując przypisanie etykiet
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    X, y = X[indices], y[indices]

    # 🔹 Normalizacja całego połączonego zbioru danych
    X = (X - np.mean(X)) / np.std(X)

    return X, y


### 🔥 **4. Tworzenie modelu CNN+LSTM**
def build_cnn_lstm(input_shape, num_classes):
    model = models.Sequential([
        layers.Conv1D(128, kernel_size=9, padding='same', activation='relu', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),
        layers.Dropout(0.2),  # 🆕 Dropout dla lepszego uogólnienia

        layers.Conv1D(256, kernel_size=7, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),
        layers.Dropout(0.3),  # 🆕

        layers.Conv1D(512, kernel_size=5, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),
        layers.Dropout(0.4),  # 🆕

        layers.LSTM(128, return_sequences=True),
        layers.Dropout(0.4),  # 🆕 Większy dropout dla lepszej generalizacji
        layers.LSTM(64, return_sequences=False),
        layers.Dropout(0.4),  # 🆕 Większy dropout dla lepszej generalizacji

        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0003),
                  loss='categorical_crossentropy',
                  metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
    return model





def train_model():
    print("Czy TensorFlow widzi GPU?", tf.config.list_physical_devices('GPU'))

    # Wczytanie danych z MITDB i SVDB
    X, y = load_all_ecg_data(MITDB_PATH, SVDB_PATH, INCARTDB_PATH)
    #X, y = balance_classes_smart(X, y)  # ✅ Nowe lepsze balansowanie klas

    X, y = balance_classes(X, y, class_to_reduce=0, reduction_factor=0.5)
    #X, y = balance_classes_oversampling(X, y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, stratify=y_train, random_state=42)

    X_train, X_val, X_test = X_train[..., np.newaxis], X_val[..., np.newaxis], X_test[..., np.newaxis]
    y_train, y_val, y_test = to_categorical(y_train, NUM_CLASSES), to_categorical(y_val, NUM_CLASSES), to_categorical(y_test, NUM_CLASSES)

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=3, min_lr=1e-5)

    model = build_cnn_lstm((SEGMENT_LENGTH, 1), NUM_CLASSES)
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=5, batch_size=256, callbacks=[early_stopping, reduce_lr])

    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_test, axis=1)

    report = classification_report(y_true, y_pred_classes)
    print("\n📊 Statystyki modelu:\n", report)

    plot_confusion_matrix(y_true, y_pred_classes, labels=list(set(LABEL_MAP.values())))
    model.save("ecg_superclass_classifier.h5")

if __name__ == "__main__":
    train_model()
