"""
4_entrenar.py
==============
Entrena el modelo GRU bidireccional sobre el dataset preprocesado,
con validación, callbacks de regularización/early stopping, evaluación
final sobre el set de test, y guardado del modelo entrenado.

Requiere TensorFlow instalado (recomendado: Google Colab con GPU, o un
entorno local con `pip install tensorflow`). Este entorno de trabajo no
tiene TensorFlow disponible, así que este script no pudo ejecutarse aquí;
se escribió y revisó cuidadosamente contra la API estable de Keras 3,
pero conviene correr `python 3_modelo.py` primero para confirmar el
`model.summary()` real antes de lanzar el entrenamiento completo.

Uso:
    python 4_entrenar.py
"""

import json
import os

import numpy as np
import tensorflow as tf

import config
from importlib import import_module

modelo_mod = import_module("3_modelo")  # nombre de archivo empieza con dígito


def cargar_datos():
    X_train = np.load(os.path.join(config.PROCESSED_DIR, "X_train.npy"))
    y_train = np.load(os.path.join(config.PROCESSED_DIR, "y_train.npy"))
    X_val = np.load(os.path.join(config.PROCESSED_DIR, "X_val.npy"))
    y_val = np.load(os.path.join(config.PROCESSED_DIR, "y_val.npy"))
    X_test = np.load(os.path.join(config.PROCESSED_DIR, "X_test.npy"))
    y_test = np.load(os.path.join(config.PROCESSED_DIR, "y_test.npy"))
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def calcular_class_weights(y_train_onehot: np.ndarray) -> dict:
    """Pondera clases con menos muestras para compensar un vocabulario
    desbalanceado (por ejemplo, si a una palabra le costó más conseguir
    repeticiones que a otras)."""
    conteos = y_train_onehot.sum(axis=0)
    conteos = np.maximum(conteos, 1)
    total = conteos.sum()
    n_clases = len(conteos)
    pesos = total / (n_clases * conteos)
    return {i: float(w) for i, w in enumerate(pesos)}


def main():
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = cargar_datos()
    print(f"Train: {X_train.shape}  Val: {X_val.shape}  Test: {X_test.shape}")

    model = modelo_mod.build_model()
    model = modelo_mod.compile_model(model)
    model.summary()

    class_weights = calcular_class_weights(y_train)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=config.EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=config.REDUCE_LR_FACTOR,
            patience=config.REDUCE_LR_PATIENCE,
            min_lr=1e-6,
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=config.KERAS_MODEL_PATH,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=config.BATCH_SIZE,
        epochs=config.MAX_EPOCHS,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=2,
    )

    # El ModelCheckpoint ya guardó el mejor modelo según val_accuracy, pero
    # nos aseguramos de que el objeto en memoria (con restore_best_weights)
    # también quede persistido:
    model.save(config.KERAS_MODEL_PATH)
    print(f"\nModelo guardado en: {config.KERAS_MODEL_PATH}")

    print("\n--- Evaluación en el set de TEST (nunca visto durante el entrenamiento) ---")
    test_loss, test_acc, test_top3 = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")
    print(f"Test top-3 accuracy: {test_top3:.4f}")

    # Matriz de confusión simple en texto, útil para detectar qué palabras
    # se confunden entre sí (falsos positivos) y priorizar más datos ahí.
    y_pred = model.predict(X_test, verbose=0)
    y_pred_idx = np.argmax(y_pred, axis=1)
    y_true_idx = np.argmax(y_test, axis=1)
    errores = {}
    for t, p in zip(y_true_idx, y_pred_idx):
        if t != p:
            par = (config.VOCABULARY[t], config.VOCABULARY[p])
            errores[par] = errores.get(par, 0) + 1
    if errores:
        print("\nConfusiones más comunes (real -> predicho: cantidad):")
        for (real, pred), n in sorted(errores.items(), key=lambda kv: -kv[1])[:15]:
            print(f"  {real} -> {pred}: {n}")
    else:
        print("\nSin errores en el set de test (revisar posible sobreajuste si el dataset es muy chico).")

    with open(os.path.join(config.PROCESSED_DIR, "historial_entrenamiento.json"), "w") as f:
        json.dump({k: [float(v) for v in vals] for k, vals in history.history.items()}, f, indent=2)


if __name__ == "__main__":
    main()
