import os
import wfdb
import tensorflow as tf
from scipy.signal import resample
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import  classification_report
import numpy as np
import signal
import sys
import tensorflow.keras.backend as K
import gc

from cnn_model import build_cnn_lstm
from constants import TARGET_FS, LABEL_MAP, SEGMENT_LENGTH, MITDB_PATH, SVDB_PATH, INCARTDB_PATH, NUM_CLASSES
from data_manipulation import balance_classes_smart
from filters import filter_ecg
from vizualizations import plot_ecg_segments, plot_confusion_matrix








def select_best_lead(record):
    if record.p_signal is None or not hasattr(record, 'sig_name'):
        return None  # Brak sygnału lub brak listy odprowadzeń

    leads = record.sig_name  # Lista nazw dostępnych kanałów
    print(f"Dostępne kanały: {leads}")

    # Preferowane odprowadzenia
    preferred_leads = ["MLII", "II", "ECG1"]

    for lead in preferred_leads:
        if lead in leads:
            return record.p_signal[:, leads.index(lead)]

    return None  # Jeśli nie znaleziono pasujących odprowadzeń



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



def train_model():
    print("Czy TensorFlow widzi GPU?", tf.config.list_physical_devices('GPU'))

    # Wczytanie danych z MITDB i SVDB
    X, y = load_all_ecg_data(MITDB_PATH, SVDB_PATH, INCARTDB_PATH)
    X, y = balance_classes_smart(X, y)  # ✅ Nowe lepsze balansowanie klas

    #plot_ecg_segments(X, y, LABEL_MAP)

    #X, y = balance_classes(X, y, class_to_reduce=0, reduction_factor=0.80)
    #X, y = balance_classes_oversampling(X, y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, stratify=y_train, random_state=42)

    X_train, X_val, X_test = X_train[..., np.newaxis], X_val[..., np.newaxis], X_test[..., np.newaxis]
    y_train, y_val, y_test = to_categorical(y_train, NUM_CLASSES), to_categorical(y_val, NUM_CLASSES), to_categorical(y_test, NUM_CLASSES)

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=3, min_lr=1e-5)

    model = build_cnn_lstm((SEGMENT_LENGTH, 1), NUM_CLASSES)
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=256, callbacks=[early_stopping, reduce_lr])

    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_test, axis=1)

    report = classification_report(y_true, y_pred_classes)
    print("\n📊 Statystyki modelu:\n", report)

    plot_confusion_matrix(y_true, y_pred_classes, labels=list(set(LABEL_MAP.values())))
    model.save("ecg_superclass_classifier.h5")





def cleanup_resources(signum, frame):
    print("🛑 Przerywanie... zwalniam pamięć!")
    K.clear_session()
    gc.collect()
    sys.exit(0)

signal.signal(signal.SIGINT, cleanup_resources)  # Obsługa Ctrl+C


if __name__ == "__main__":
    train_model()
