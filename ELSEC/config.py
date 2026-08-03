"""
config.py
==========
Configuración central del sistema de reconocimiento de Lengua de Señas
Ecuatoriana (LSEC). Todos los scripts (captura, preprocesamiento,
entrenamiento, conversión) importan de aquí para garantizar consistencia
en todo el pipeline.
"""

import os

# ---------------------------------------------------------------------------
# 1. VOCABULARIO INICIAL (50 palabras)
# ---------------------------------------------------------------------------
# IMPORTANTE: esta lista propone QUÉ palabras conviene priorizar por su alta
# utilidad comunicativa (criterio de diseño de producto), pero Claude no
# puede verificar ni enseñar la seña (forma manual) correcta de cada una.
# La forma visual de cada seña debe tomarse del Diccionario de Lengua de
# Señas Ecuatoriana "Gabriel Román" (FENASEC / CONADIS / U. Tecnológica
# Indoamérica) o, idealmente, capturarse directamente con intérpretes
# certificados o personas Sordas señantes de LSEC. Ver README.md sección 8.
VOCABULARY = [
    # Saludos y cortesía (8)
    "HOLA", "ADIOS", "GRACIAS", "POR_FAVOR", "DE_NADA",
    "BUENOS_DIAS", "BUENAS_NOCHES", "DISCULPA",
    # Familia (7)
    "MAMA", "PAPA", "HERMANO", "HERMANA", "ABUELO", "ABUELA", "HIJO",
    # Pronombres y preguntas (7)
    "YO", "TU", "QUE", "QUIEN", "DONDE", "CUANDO", "COMO",
    # Verbos / necesidades básicas (10)
    "COMER", "BEBER", "DORMIR", "AYUDAR", "QUERER",
    "IR", "VENIR", "TRABAJAR", "ESTUDIAR", "JUGAR",
    # Emociones y estados (6)
    "FELIZ", "TRISTE", "ENOJADO", "BIEN", "MAL", "MIEDO",
    # Lugares y objetos comunes (6)
    "CASA", "BANO", "ESCUELA", "AGUA", "COMIDA", "DINERO",
    # Colores (4)
    "ROJO", "AZUL", "VERDE", "AMARILLO",
    # Afirmación / negación (2)
    "SI", "NO",
]
assert len(VOCABULARY) == 50, f"Se esperaban 50 palabras, hay {len(VOCABULARY)}"
NUM_CLASSES = len(VOCABULARY)

# ---------------------------------------------------------------------------
# 2. LANDMARKS DE MEDIAPIPE A UTILIZAR
# ---------------------------------------------------------------------------
# Pose: usamos solo el tren superior (nariz, hombros, codos, muñecas y los
# puntos de mano-desde-pose). Se descartan cadera/piernas/pies (índices
# 23-32) porque no aportan información para el reconocimiento de señas y
# solo añaden ruido/varianza al modelo.
POSE_LANDMARKS_IDX = [0, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]  # 13 puntos
N_POSE = len(POSE_LANDMARKS_IDX)          # 13 landmarks x (x,y,z,visibility) = 52
N_HAND = 21                                # 21 landmarks x (x,y,z) = 63 por mano

POSE_FEATS = N_POSE * 4                    # 52
HAND_FEATS = N_HAND * 3                    # 63
NUM_FEATURES = POSE_FEATS + HAND_FEATS * 2  # 52 + 63 + 63 = 178

# Índices (dentro del subconjunto POSE_LANDMARKS_IDX) de hombro izq/der,
# usados como referencia para normalizar (centro=punto medio de hombros).
LEFT_SHOULDER_SUBIDX = POSE_LANDMARKS_IDX.index(11)
RIGHT_SHOULDER_SUBIDX = POSE_LANDMARKS_IDX.index(12)

# ---------------------------------------------------------------------------
# 3. SECUENCIA TEMPORAL
# ---------------------------------------------------------------------------
SEQUENCE_LENGTH = 45     # frames por secuencia tras remuestreo (~1.5-2.2s de seña)
MIN_RAW_FRAMES = 10      # descartar capturas demasiado cortas/corruptas

# ---------------------------------------------------------------------------
# 4. HIPERPARÁMETROS DE ENTRENAMIENTO
# ---------------------------------------------------------------------------
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
MAX_EPOCHS = 200                 # tope superior; EarlyStopping cortará antes
EARLY_STOPPING_PATIENCE = 20
REDUCE_LR_PATIENCE = 7
REDUCE_LR_FACTOR = 0.5
DROPOUT_RATE_LOW = 0.3
DROPOUT_RATE_HIGH = 0.4
L2_REG = 1e-3
LABEL_SMOOTHING = 0.1
OPTIMIZER_NAME = "adam"

# Split del dataset
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Muestras recomendadas por palabra (ver README.md sección 6)
MIN_SAMPLES_PER_WORD = 50        # mínimo viable
RECOMMENDED_SAMPLES_PER_WORD = 100  # recomendado para producción
MIN_SIGNERS_PER_WORD = 3         # mínimo de personas distintas por palabra

# ---------------------------------------------------------------------------
# 5. RUTAS
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
RAW_VIDEOS_DIR = os.path.join(DATASET_DIR, "raw_videos")
KEYPOINTS_DIR = os.path.join(DATASET_DIR, "keypoints")
PROCESSED_DIR = os.path.join(DATASET_DIR, "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")

MEDIAPIPE_MODELS_DIR = os.path.join(BASE_DIR, "mp_models")
POSE_MODEL_PATH = os.path.join(MEDIAPIPE_MODELS_DIR, "pose_landmarker_lite.task")
HAND_MODEL_PATH = os.path.join(MEDIAPIPE_MODELS_DIR, "hand_landmarker.task")

KERAS_MODEL_PATH = os.path.join(MODELS_DIR, "lsec_model.keras")
TFLITE_MODEL_PATH = os.path.join(MODELS_DIR, "lsec_model.tflite")
TFLITE_MODEL_QUANT_PATH = os.path.join(MODELS_DIR, "lsec_model_int8.tflite")
LABELS_PATH = os.path.join(MODELS_DIR, "labels.txt")
LABEL_MAP_PATH = os.path.join(PROCESSED_DIR, "label_map.json")

for _d in [DATASET_DIR, RAW_VIDEOS_DIR, KEYPOINTS_DIR, PROCESSED_DIR,
           MODELS_DIR, MEDIAPIPE_MODELS_DIR]:
    os.makedirs(_d, exist_ok=True)
