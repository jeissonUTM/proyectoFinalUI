"""
2_preprocesar_dataset.py
==========================
Convierte las secuencias crudas de keypoints (dataset/keypoints/<PALABRA>/*.npy,
de longitud variable) en un dataset final de tamaño fijo listo para
entrenar: normaliza cada frame, remuestrea cada secuencia a
config.SEQUENCE_LENGTH frames, arma splits estratificados train/val/test
y guarda todo en dataset/processed/.

Pasos de preparación de datos (ver también README.md sección 7):
  1. Normalización espacial: se centra cada frame restando el punto medio
     entre hombros y se escala dividiendo por el ancho de hombros. Esto
     hace que el modelo sea invariante a la distancia a la cámara, el
     tamaño corporal de la persona y su posición en el encuadre.
  2. Remuestreo temporal: en vez de recortar/rellenar con ceros (que
     desperdicia información o introduce padding artificial), se usa
     interpolación lineal para llevar CUALQUIER secuencia cruda (p.ej. 22,
     38 o 70 frames, según la velocidad al señar) a exactamente
     SEQUENCE_LENGTH frames, preservando la forma temporal completa del
     gesto. Como resultado, el modelo no necesita una capa Masking.
  3. División estratificada: se separa por palabra (clase) para que
     train/val/test mantengan la misma proporción de cada palabra.
"""

import glob
import json
import os

import numpy as np

import config


def normalizar_frame(frame_178: np.ndarray) -> np.ndarray:
    """Centra en el punto medio de hombros y escala por el ancho de hombros."""
    pose = frame_178[: config.POSE_FEATS].reshape(config.N_POSE, 4).copy()
    left_hand = frame_178[config.POSE_FEATS: config.POSE_FEATS + config.HAND_FEATS].reshape(21, 3).copy()
    right_hand = frame_178[config.POSE_FEATS + config.HAND_FEATS:].reshape(21, 3).copy()

    l_shoulder = pose[config.LEFT_SHOULDER_SUBIDX, :3]
    r_shoulder = pose[config.RIGHT_SHOULDER_SUBIDX, :3]
    center = (l_shoulder + r_shoulder) / 2.0
    scale = np.linalg.norm(l_shoulder - r_shoulder) + 1e-6

    # Si no se detectó torso en el frame (todo cero), no normalizamos para
    # evitar dividir por ~0 y generar valores absurdos; se deja en cero.
    if scale < 1e-4:
        return frame_178

    pose[:, :3] = (pose[:, :3] - center) / scale
    left_hand = np.where(left_hand.any(axis=1, keepdims=True), (left_hand - center) / scale, left_hand)
    right_hand = np.where(right_hand.any(axis=1, keepdims=True), (right_hand - center) / scale, right_hand)

    return np.concatenate([pose.flatten(), left_hand.flatten(), right_hand.flatten()]).astype(np.float32)


def remuestrear_secuencia(seq: np.ndarray, target_len: int) -> np.ndarray:
    """Interpola linealmente en el eje temporal para llevar (T, F) -> (target_len, F)."""
    original_len = seq.shape[0]
    if original_len == target_len:
        return seq
    original_idx = np.linspace(0.0, 1.0, original_len)
    target_idx = np.linspace(0.0, 1.0, target_len)
    out = np.empty((target_len, seq.shape[1]), dtype=np.float32)
    for f in range(seq.shape[1]):
        out[:, f] = np.interp(target_idx, original_idx, seq[:, f])
    return out


def procesar_secuencia_completa(seq_cruda: np.ndarray) -> np.ndarray:
    seq_norm = np.stack([normalizar_frame(f) for f in seq_cruda])
    return remuestrear_secuencia(seq_norm, config.SEQUENCE_LENGTH)


def cargar_dataset_crudo():
    X, y = [], []
    conteo = {}
    for palabra in config.VOCABULARY:
        carpeta = os.path.join(config.KEYPOINTS_DIR, palabra)
        archivos = sorted(glob.glob(os.path.join(carpeta, "*.npy")))
        conteo[palabra] = len(archivos)
        for f in archivos:
            seq_cruda = np.load(f)
            if seq_cruda.shape[0] < config.MIN_RAW_FRAMES:
                continue
            X.append(procesar_secuencia_completa(seq_cruda))
            y.append(palabra)

    print("\n--- Muestras por palabra ---")
    faltantes = []
    for palabra, n in conteo.items():
        marca = "OK" if n >= config.MIN_SAMPLES_PER_WORD else "BAJO"
        print(f"  {palabra:15s}: {n:4d}  [{marca}]")
        if n < config.MIN_SAMPLES_PER_WORD:
            faltantes.append(palabra)
    if faltantes:
        print(f"\n[AVISO] {len(faltantes)} palabra(s) por debajo del mínimo recomendado "
              f"({config.MIN_SAMPLES_PER_WORD} muestras): {faltantes}")

    return np.array(X, dtype=np.float32), np.array(y)


def split_estratificado(X, y, seed=42):
    """Split train/val/test manteniendo la proporción de cada clase,
    sin depender de scikit-learn (usa solo numpy)."""
    rng = np.random.default_rng(seed)
    idx_train, idx_val, idx_test = [], [], []

    for palabra in config.VOCABULARY:
        idx_clase = np.where(y == palabra)[0]
        rng.shuffle(idx_clase)
        n = len(idx_clase)
        n_train = int(round(n * config.TRAIN_RATIO))
        n_val = int(round(n * config.VAL_RATIO))
        idx_train.extend(idx_clase[:n_train])
        idx_val.extend(idx_clase[n_train:n_train + n_val])
        idx_test.extend(idx_clase[n_train + n_val:])

    idx_train, idx_val, idx_test = map(np.array, (idx_train, idx_val, idx_test))
    rng.shuffle(idx_train)
    rng.shuffle(idx_val)
    rng.shuffle(idx_test)
    return (X[idx_train], y[idx_train]), (X[idx_val], y[idx_val]), (X[idx_test], y[idx_test])


def one_hot(y_labels, label_to_idx):
    idx = np.array([label_to_idx[l] for l in y_labels])
    out = np.zeros((len(idx), config.NUM_CLASSES), dtype=np.float32)
    out[np.arange(len(idx)), idx] = 1.0
    return out


def main():
    print("Cargando y procesando secuencias crudas...")
    X, y = cargar_dataset_crudo()
    print(f"\nTotal de secuencias válidas: {len(X)}")
    if len(X) == 0:
        print("No hay datos para procesar. Ejecuta primero 1_capturar_keypoints.py")
        return

    label_to_idx = {palabra: i for i, palabra in enumerate(config.VOCABULARY)}

    (X_train, y_train), (X_val, y_val), (X_test, y_test) = split_estratificado(X, y)

    y_train_oh = one_hot(y_train, label_to_idx)
    y_val_oh = one_hot(y_val, label_to_idx)
    y_test_oh = one_hot(y_test, label_to_idx)

    np.save(os.path.join(config.PROCESSED_DIR, "X_train.npy"), X_train)
    np.save(os.path.join(config.PROCESSED_DIR, "y_train.npy"), y_train_oh)
    np.save(os.path.join(config.PROCESSED_DIR, "X_val.npy"), X_val)
    np.save(os.path.join(config.PROCESSED_DIR, "y_val.npy"), y_val_oh)
    np.save(os.path.join(config.PROCESSED_DIR, "X_test.npy"), X_test)
    np.save(os.path.join(config.PROCESSED_DIR, "y_test.npy"), y_test_oh)

    with open(config.LABEL_MAP_PATH, "w", encoding="utf-8") as f:
        json.dump(label_to_idx, f, ensure_ascii=False, indent=2)

    print(f"\nTrain: {X_train.shape}  Val: {X_val.shape}  Test: {X_test.shape}")
    print(f"Guardado en: {config.PROCESSED_DIR}")


if __name__ == "__main__":
    main()
