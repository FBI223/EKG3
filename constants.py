

# 📂 Foldery z danymi MITDB i SVDB
MITDB_PATH = "mitdb/"
SVDB_PATH = "svdb/"
INCARTDB_PATH = "incartdb/"

# 🔹 Docelowa częstotliwość próbkowania
TARGET_FS = 360
SEGMENT_LENGTH = 300   # Długość segmentu w próbkach (QRS w środku)

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

