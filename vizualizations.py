import os

import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(y_true, y_pred, labels):
    """Rysuje macierz błędów."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()


def plot_ecg_segments(X, y, label_map, samples_per_class=200, save_dir="temp"):
    """
    Zapisuje `samples_per_class` losowych wykresów dla każdej podklasy w `LABEL_MAP` do folderu `temp`,
    jeśli w podklasie jest mniej próbek, zapisuje wszystkie dostępne.
    """
    os.makedirs(save_dir, exist_ok=True)

    subclass_indices = {subclass: np.where(y == label_map[subclass])[0] for subclass in label_map.keys()}

    for subclass, indices in subclass_indices.items():
        num_samples = min(samples_per_class, len(indices))
        if num_samples == 0:
            continue  # Pomijamy podklasy, które nie mają żadnych danych

        selected_indices = np.random.choice(indices, num_samples, replace=False)

        safe_subclass = subclass.replace("/", "slash").replace("?", "unknown")

        for i, idx in enumerate(selected_indices):
            plt.figure(figsize=(6, 3))
            plt.plot(X[idx], label=f"Subclass {subclass} (Class {label_map[subclass]})")
            plt.title(f"ECG Segment - Subclass {subclass}")
            plt.xlabel("Sample Index")
            plt.ylabel("Amplitude")
            plt.legend()
            save_path = os.path.join(save_dir, f"{safe_subclass}_{i}.png").replace("\\", "/")
            plt.savefig(save_path)
            plt.close()
