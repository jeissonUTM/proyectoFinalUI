"""
3_modelo.py
============
Arquitectura recomendada: pila de LSTM bidireccionales.

CAMBIO IMPORTANTE (corregido tras error de conversión a TFLite):
Este script usaba originalmente GRU. Se cambió a LSTM por una razón
puramente técnica de despliegue, no de precisión:

  TFLite NO tiene un kernel nativo fusionado para GRU (solo para LSTM
  y SimpleRNN). Al convertir un modelo con GRU, el conversor no puede
  fusionarlo y lo descompone en un bucle manual con operadores
  TensorListReserve/SetItem/Stack, que NO son built-ins de TFLite y
  requieren el delegado Flex (SELECT_TF_OPS) para ejecutarse. Eso
  produce un .tflite que:
    - Falla en tf.lite.Interpreter estándar sin el delegado Flex
      enlazado ("RuntimeError: Select TensorFlow op(s)... not
      supported by this interpreter").
    - Es inviable en el frontend Flutter con `tflite_flutter`, que no
      trae el delegado Flex integrado (habría que compilar un binario
      nativo custom, y en Android sumaría varios MB de
      tensorflow-lite-select-tf-ops).

LSTM, en cambio, SÍ tiene conversión nativa a los ops
UnidirectionalSequenceLSTM / BidirectionalSequenceLSTM, así que el
.tflite resultante corre con el intérprete liviano estándar, sin Flex,
tal como necesita `tflite_flutter`. El costo es ~25-30% más parámetros
que GRU (LSTM tiene 4 compuertas + estado de celda separado), pero con
las unidades usadas acá (128/64/32) el modelo sigue siendo muy chico
para 50 clases.

Por qué no Transformer (sigue aplicando el mismo razonamiento):
Con un dataset de tamaño moderado (decenas de miles de secuencias como
mucho para 50 palabras), un Transformer necesita bastante más datos y
cómputo para igualar a un LSTM bien regularizado; su ventaja (atención
global) importa más en secuencias largas, y aquí cada seña son solo
~45 frames. Para reconocimiento de palabras aisladas ya tenemos la
ventana temporal COMPLETA antes de clasificar, así que usamos LSTM
Bidireccional (lee la secuencia en ambos sentidos), lo cual mejora la
precisión sin costo en producción (no es streaming token-a-token).

Arquitectura:

  Input (45, 178)
  -> GaussianNoise(0.01)                          [solo activo en training]
  -> Bidirectional(LSTM(128, return_sequences=True))
  -> BatchNormalization -> Dropout(0.3)
  -> Bidirectional(LSTM(64,  return_sequences=True))
  -> BatchNormalization -> Dropout(0.3)
  -> Bidirectional(LSTM(32,  return_sequences=False))
  -> BatchNormalization -> Dropout(0.4)
  -> Dense(64, relu, L2=1e-3)
  -> Dropout(0.3)
  -> Dense(50, softmax)

IMPORTANTE: si ya habías entrenado un modelo con la versión GRU
anterior, ese .keras/.tflite NO sirve para esta arquitectura nueva.
Hay que volver a correr 4_entrenar.py (y después 5_convertir_tflite.py)
desde cero.

NOTA SOBRE batch_size (fix de un segundo error de conversión):
El conversor de TFLite solo puede fusionar un LSTM bidireccional en un
op nativo (BidirectionalSequenceLSTM) cuando el tamaño de batch del
grafo es ESTÁTICO. Con `shape=(sequence_length, num_features)` Keras
deja el batch implícito en `None` (dinámico), lo cual es correcto y
necesario para entrenar con distintos batch_size, pero rompe la
conversión del LSTM hacia atrás con
`'tf.TensorListReserve' op requires element_shape to be static`.
Por eso `build_model()` ahora acepta un parámetro `batch_size`:
  - `batch_size=None` (default): para entrenar (4_entrenar.py). Batch
    dinámico, como siempre.
  - `batch_size=1`: usado SOLO por 5_convertir_tflite.py, que reconstruye
    el modelo con este batch fijo, le copia los pesos ya entrenados, y
    recién ahí convierte. No requiere reentrenar nada.
"""

import tensorflow as tf
from tensorflow.keras import layers, regularizers, Model
from typing import Optional

import config


def build_model(
    sequence_length: int = config.SEQUENCE_LENGTH,
    num_features: int = config.NUM_FEATURES,
    num_classes: int = config.NUM_CLASSES,
    batch_size: Optional[int] = None,
) -> Model:
    if batch_size is None:
        inputs = layers.Input(shape=(sequence_length, num_features), name="keypoints")
    else:
        # batch_shape fija el tamaño de batch en el grafo (necesario para
        # que TFLite pueda fusionar el LSTM bidireccional). Se usa
        # exclusivamente al convertir a TFLite, con batch_size=1, ya que
        # en el celular las secuencias se procesan de a una por vez.
        inputs = layers.Input(
            batch_shape=(batch_size, sequence_length, num_features), name="keypoints"
        )

    x = layers.GaussianNoise(0.01)(inputs)  # pequeño ruido: regulariza y simula jitter de MediaPipe

    # NOTA: activation="tanh" + recurrent_activation="sigmoid" + unroll=False
    # (default) son justo la combinación que el conversor de TFLite sabe
    # fusionar de forma nativa para LSTM. No cambiar estos parámetros sin
    # volver a probar la conversión.
    x = layers.Bidirectional(
        layers.LSTM(128, return_sequences=True, activation="tanh", recurrent_activation="sigmoid")
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(config.DROPOUT_RATE_LOW)(x)

    x = layers.Bidirectional(
        layers.LSTM(64, return_sequences=True, activation="tanh", recurrent_activation="sigmoid")
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(config.DROPOUT_RATE_LOW)(x)

    x = layers.Bidirectional(
        layers.LSTM(32, return_sequences=False, activation="tanh", recurrent_activation="sigmoid")
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(config.DROPOUT_RATE_HIGH)(x)

    x = layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l2(config.L2_REG))(x)
    x = layers.Dropout(config.DROPOUT_RATE_LOW)(x)

    outputs = layers.Dense(num_classes, activation="softmax", name="palabra")(x)

    model = Model(inputs, outputs, name="lsec_lstm_bidireccional")
    return model


def compile_model(model: Model) -> Model:
    optimizer = tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE)
    loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing=config.LABEL_SMOOTHING)
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=["accuracy", tf.keras.metrics.TopKCategoricalAccuracy(k=3, name="top3_accuracy")],
    )
    return model


if __name__ == "__main__":
    m = build_model()
    m = compile_model(m)
    m.summary()
    total_params = m.count_params()
    print(f"\nParámetros totales: {total_params:,}")
    print(f"Tamaño estimado sin comprimir (float32): {total_params * 4 / 1024:.1f} KB")
