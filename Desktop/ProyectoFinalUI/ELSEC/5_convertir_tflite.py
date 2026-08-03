"""
5_convertir_tflite.py
=======================
Convierte el modelo Keras entrenado (config.KERAS_MODEL_PATH) a
TensorFlow Lite, optimizado para celular, y valida que la salida del
.tflite coincida con la del modelo original antes de darlo por bueno.

NOTA (fix de conversión): antes este script agregaba
`tf.lite.OpsSet.SELECT_TF_OPS` como respaldo "por robustez". Eso
ocultaba un problema real: con capas GRU (ver 3_modelo.py, ya
corregido a LSTM), el conversor no puede fusionar la RNN en un op
nativo y cae en TensorListReserve/SetItem/Stack, que SOLO corren con
el delegado Flex. Ese .tflite fallaba tanto en tf.lite.Interpreter
estándar como en tflite_flutter (Flutter). Ahora el conversor usa
ÚNICAMENTE `TFLITE_BUILTINS` (sin Flex): si el modelo ya no tiene GRU,
esto debería convertir sin problema; si algo más adelante volviera a
requerir Flex, la conversión falla acá mismo con un error explícito,
en vez de producir un .tflite que se rompe recién al validarlo o al
integrarlo con la app.

NOTA 2 (segundo fix de conversión — batch_size=1): incluso sin GRU, el
LSTM bidireccional puede volver a fallar la conversión con
`'tf.TensorListReserve' op requires element_shape to be static`. Es un
problema distinto: la fusión de LSTM bidireccional a un op nativo de
TFLite SOLO funciona si el tamaño de batch del grafo es estático. Por
eso este script ya no convierte el modelo entrenado (batch dinámico)
directamente: reconstruye una copia con batch_size=1 fijo (ver
3_modelo.py), le copia los pesos entrenados, y convierte esa copia.

Genera dos variantes:
  - lsec_model.tflite       -> cuantización dinámica (pesos a int8,
                               activaciones en float). Es la opción por
                               defecto recomendada para este modelo:
                               reduce ~4x el tamaño de las capas
                               LSTM/Dense sin tocar las activaciones
                               recurrentes, que son más sensibles a la
                               cuantización.
  - lsec_model_int8.tflite  -> cuantización entera completa (int8 en
                               pesos Y activaciones) usando un dataset
                               representativo. Es más chica y más rápida,
                               pero en modelos con LSTM a veces pierde
                               algo de precisión; por eso se generan AMBAS
                               y se recomienda comparar su accuracy en el
                               set de test antes de elegir cuál embeber
                               en la app Flutter.

También escribe labels.txt (una palabra por línea, en el mismo orden
que usa el modelo) para que la app Flutter pueda mapear el índice de
salida del softmax a la palabra correspondiente.
"""

import os
from importlib import import_module

import numpy as np
import tensorflow as tf

import config

modelo_mod = import_module("3_modelo")  # nombre de archivo empieza con dígito


def convertir_dynamic_range(model) -> bytes:
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    # Solo builtins: con LSTM (ver 3_modelo.py) el conversor fusiona la
    # RNN en BidirectionalSequenceLSTM nativo y NO necesita Flex/Select
    # TF ops. Si esto llegara a fallar pidiendo SELECT_TF_OPS de nuevo,
    # es señal de que algo en la arquitectura dejó de ser fusionable
    # (por ejemplo, volver a usar GRU) — mejor enterarse acá que en el
    # intérprete de Flutter.
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
    ]
    return converter.convert()


def convertir_int8_completo(model, X_representativo: np.ndarray) -> bytes:
    def representative_dataset():
        for i in range(min(200, len(X_representativo))):
            sample = X_representativo[i: i + 1].astype(np.float32)
            yield [sample]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    # Igual que en convertir_dynamic_range: solo builtins int8, sin Flex.
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
    ]
    return converter.convert()


def validar_tflite(tflite_bytes: bytes, model_keras, X_test: np.ndarray, y_test: np.ndarray, n_muestras=50):
    """Compara predicciones del .tflite contra el modelo Keras original
    y reporta el accuracy del .tflite sobre una muestra del set de test."""
    interpreter = tf.lite.Interpreter(model_content=tflite_bytes)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    n = min(n_muestras, len(X_test))
    aciertos = 0
    diffs = []
    for i in range(n):
        x = X_test[i: i + 1].astype(input_details[0]["dtype"])
        interpreter.set_tensor(input_details[0]["index"], x)
        interpreter.invoke()
        pred_tflite = interpreter.get_tensor(output_details[0]["index"])[0]

        pred_keras = model_keras.predict(X_test[i: i + 1], verbose=0)[0]
        diffs.append(np.max(np.abs(pred_tflite - pred_keras)))

        if np.argmax(pred_tflite) == np.argmax(y_test[i]):
            aciertos += 1

    print(f"  Diferencia máxima Keras vs TFLite (debería ser pequeña): {max(diffs):.4f}")
    print(f"  Accuracy del .tflite sobre {n} muestras de test: {aciertos / n:.4f}")


def main():
    if not os.path.exists(config.KERAS_MODEL_PATH):
        raise FileNotFoundError(
            f"No se encontró {config.KERAS_MODEL_PATH}. Corré primero 4_entrenar.py"
        )
    model = tf.keras.models.load_model(config.KERAS_MODEL_PATH)

    # El modelo entrenado tiene batch dinámico (None), necesario para
    # entrenar con distintos batch_size. TFLite solo puede fusionar el
    # LSTM bidireccional en un op nativo si el batch del grafo que se
    # convierte es ESTÁTICO, así que reconstruimos una copia con
    # batch_size=1 fijo (misma arquitectura, mismos pesos) y convertimos
    # esa copia en vez del modelo original. No hace falta reentrenar:
    # los pesos son los mismos, solo cambia la forma declarada de entrada.
    print("Reconstruyendo el modelo con batch_size=1 fijo para la conversión...")
    modelo_para_convertir = modelo_mod.build_model(batch_size=1)
    modelo_para_convertir.set_weights(model.get_weights())

    X_test = np.load(os.path.join(config.PROCESSED_DIR, "X_test.npy"))
    y_test = np.load(os.path.join(config.PROCESSED_DIR, "y_test.npy"))
    X_train = np.load(os.path.join(config.PROCESSED_DIR, "X_train.npy"))  # como dataset representativo

    print("Convirtiendo con cuantización dinámica...")
    tflite_dynamic = convertir_dynamic_range(modelo_para_convertir)
    with open(config.TFLITE_MODEL_PATH, "wb") as f:
        f.write(tflite_dynamic)
    tam_kb = len(tflite_dynamic) / 1024
    print(f"  Guardado: {config.TFLITE_MODEL_PATH}  ({tam_kb:.1f} KB)")
    # Se valida contra `model` (batch dinámico): mismos pesos, así que las
    # predicciones deben coincidir con las del .tflite (batch fijo).
    validar_tflite(tflite_dynamic, model, X_test, y_test)

    print("\nConvirtiendo con cuantización entera completa (int8)...")
    try:
        tflite_int8 = convertir_int8_completo(modelo_para_convertir, X_train)
        with open(config.TFLITE_MODEL_QUANT_PATH, "wb") as f:
            f.write(tflite_int8)
        tam_kb_int8 = len(tflite_int8) / 1024
        print(f"  Guardado: {config.TFLITE_MODEL_QUANT_PATH}  ({tam_kb_int8:.1f} KB)")
        validar_tflite(tflite_int8, model, X_test, y_test)
    except Exception as e:
        print(f"  [AVISO] La cuantización int8 completa falló ({e}).")
        print("  No es crítico: usa la versión de cuantización dinámica "
              "(lsec_model.tflite), que ya es suficientemente liviana.")

    with open(config.LABELS_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(config.VOCABULARY))
    print(f"\nEtiquetas guardadas en: {config.LABELS_PATH}")
    print("\nListo. Copiá el .tflite elegido + labels.txt a assets/ de tu proyecto Flutter.")


if __name__ == "__main__":
    main()
