"""
4b_probar_en_vivo.py
======================
Paso intermedio entre 4_entrenar.py y 5_convertir_tflite.py: prueba el
modelo Keras recién entrenado (models/lsec_model.keras) en vivo por
webcam, ANTES de invertir tiempo en convertirlo a TFLite. Inspirado en
el flujo de evaluate_model.py (detección continua por presencia de
mano + umbral de confianza + texto a voz), pero adaptado a este
proyecto:

  - Usa PoseLandmarker + HandLandmarker (ExtractorKeypoints, definido en
    1_capturar_keypoints.py) en vez de la Holistic legacy, que ya
    comprobamos que no existe en las versiones actuales de mediapipe.
  - Reutiliza normalizar_frame() y remuestrear_secuencia() de
    2_preprocesar_dataset.py, así el preprocesamiento en inferencia es
    IDÉNTICO al que se usó para entrenar (una fuente muy común de bugs
    silenciosos es normalizar distinto en train y en inferencia).
  - Carga el modelo .keras directo (no hace falta tener el .tflite
    todavía: eso es el paso 5, y conviene validar el modelo ANTES de
    convertirlo, tal como pediste).

Cómo decide dónde empieza y termina una seña (clase SegmentadorSenas):
mientras detecta al menos una mano, va acumulando frames. Cuando deja
de ver manos, tolera un hueco corto (frames_espera) por si fue una
oclusión momentánea; si el hueco se extiende, cierra la secuencia con
los frames acumulados HASTA ANTES del hueco (no mete frames “vacíos”
en la seña) y la manda al modelo. Si la seña detectada fue demasiado
corta, la descarta como ruido en vez de intentar clasificarla (así se
evitan falsos positivos por gestos accidentales).

Uso:
    python 4b_probar_en_vivo.py
    python 4b_probar_en_vivo.py --umbral 0.85 --camara 0
    python 4b_probar_en_vivo.py --sin-voz     # desactiva el texto a voz
"""

import argparse
import queue
import threading
import time
from importlib import import_module

import cv2
import numpy as np
import tensorflow as tf

import config

captura_mod = import_module("1_capturar_keypoints")
preproc_mod = import_module("2_preprocesar_dataset")
ExtractorKeypoints = captura_mod.ExtractorKeypoints
VisionRunningMode = captura_mod.VisionRunningMode
normalizar_frame = preproc_mod.normalizar_frame
remuestrear_secuencia = preproc_mod.remuestrear_secuencia


class SegmentadorSenas:
    """Aísla una seña dentro de un flujo continuo de frames, en base a
    si hay mano(s) detectada(s) o no. Es lógica pura (sin cámara ni
    MediaPipe adentro), así que se puede probar por separado."""

    def __init__(self, min_frames=config.MIN_RAW_FRAMES, frames_espera=6):
        self.buffer = []
        self.frames_sin_mano = 0
        self.min_frames = min_frames
        self.frames_espera = frames_espera

    def procesar(self, vec: np.ndarray, mano_detectada: bool):
        """Devuelve un array (T, 178) cuando una seña recién terminó,
        o None si todavía está en curso / no hay nada que reportar."""
        if mano_detectada:
            self.buffer.append(vec)
            self.frames_sin_mano = 0
            return None

        if not self.buffer:
            return None  # no se está grabando nada, no hay que hacer nada

        self.frames_sin_mano += 1
        if self.frames_sin_mano < self.frames_espera:
            return None  # hueco corto: tolerar, no cortar todavía

        # el hueco se extendió: cerrar la secuencia tal como estaba
        # ANTES de que empezara el hueco (por eso no se le agregaron
        # frames sin mano al buffer mientras se esperaba)
        seq = np.array(self.buffer, dtype=np.float32)
        self.buffer = []
        self.frames_sin_mano = 0
        if len(seq) < self.min_frames:
            return None  # muy corta: probablemente ruido, se descarta
        return seq


class HabladorEnSegundoPlano:
    """Envoltorio simple sobre pyttsx3 que habla en un hilo aparte para
    no congelar el video mientras se reproduce el audio. Si pyttsx3 no
    está instalado, se degrada a solo imprimir en consola (no rompe el
    resto del script)."""

    def __init__(self, activo=True):
        self.activo = activo
        self.cola = queue.Queue()
        self.engine = None
        if activo:
            try:
                import pyttsx3
                self.engine = pyttsx3.init()
            except Exception as e:
                print(f"[AVISO] pyttsx3 no disponible ({e}). "
                      f"Instalá con: pip install pyttsx3. Continuando sin voz.")
                self.activo = False
        if self.activo:
            self.hilo = threading.Thread(target=self._loop, daemon=True)
            self.hilo.start()

    def _loop(self):
        while True:
            texto = self.cola.get()
            if texto is None:
                break
            self.engine.say(texto)
            self.engine.runAndWait()

    def decir(self, texto: str):
        if self.activo:
            self.cola.put(texto)
        else:
            print(f"[voz desactivada] {texto}")


def preprocesar_para_modelo(seq_cruda: np.ndarray) -> np.ndarray:
    """Aplica EXACTAMENTE los mismos pasos que 2_preprocesar_dataset.py
    (normalizar cada frame + remuestrear a SEQUENCE_LENGTH), para que la
    entrada al modelo en inferencia coincida con la del entrenamiento."""
    seq_norm = np.stack([normalizar_frame(f) for f in seq_cruda])
    return remuestrear_secuencia(seq_norm, config.SEQUENCE_LENGTH)


def main():
    parser = argparse.ArgumentParser(description="Prueba en vivo del modelo LSEC (.keras)")
    parser.add_argument("--umbral", type=float, default=0.80,
                         help="Confianza mínima del softmax para aceptar una predicción (default 0.80)")
    """ la esta en defaul"""
    parser.add_argument("--camara", type=int, default=0)
    parser.add_argument("--frames-espera", type=int, default=6,
                         help="Frames sin mano tolerados antes de cerrar la seña")
    parser.add_argument("--sin-voz", action="store_true", help="Desactiva el texto a voz")
    parser.add_argument("--cooldown", type=float, default=2.0,
                         help="Segundos mínimos antes de volver a anunciar la MISMA palabra")
    args = parser.parse_args()

    print(f"Cargando modelo desde {config.KERAS_MODEL_PATH} ...")
    model = tf.keras.models.load_model(config.KERAS_MODEL_PATH)

    hablador = HabladorEnSegundoPlano(activo=not args.sin_voz)
    segmentador = SegmentadorSenas(frames_espera=args.frames_espera)

    oraciones = []          # historial de palabras reconocidas (para mostrar en pantalla)
    ultima_palabra = None
    ultima_vez = 0.0

    with ExtractorKeypoints(running_mode=VisionRunningMode.VIDEO) as extractor:
        cap = cv2.VideoCapture(args.camara)
        t0 = time.time()

        while True:
            ok, frame = cap.read()
            if not ok:
                print("No se pudo leer la cámara.")
                break

            frame = cv2.flip(frame, 1)
            timestamp_ms = int((time.time() - t0) * 1000)
            vec = extractor.process_bgr_frame(frame, timestamp_ms)

            # Inferimos presencia de mano a partir del propio vector: la
            # porción de pose siempre está (o en ceros si no hay
            # persona), pero la porción de manos queda toda en cero solo
            # cuando NINGUNA mano fue detectada en el frame.
            porcion_manos = vec[config.POSE_FEATS:]
            mano_detectada = bool(np.any(porcion_manos != 0))

            resultado = segmentador.procesar(vec, mano_detectada)

            if resultado is not None:
                entrada = preprocesar_para_modelo(resultado)
                probs = model.predict(entrada[np.newaxis, ...], verbose=0)[0]
                idx = int(np.argmax(probs))
                confianza = float(probs[idx])
                palabra = config.VOCABULARY[idx]

                print(f"  -> {palabra}  ({confianza*100:.1f}%)  [{resultado.shape[0]} frames crudos]")

                ahora = time.time()
                es_repetida_reciente = (palabra == ultima_palabra) and (ahora - ultima_vez < args.cooldown)

                if confianza >= args.umbral and not es_repetida_reciente:
                    oraciones.insert(0, palabra)
                    oraciones = oraciones[:6]
                    hablador.decir(palabra)
                    ultima_palabra = palabra
                    ultima_vez = ahora

            # --- overlay en pantalla ---
            cv2.rectangle(frame, (0, 0), (frame.shape[1], 40), (245, 117, 16), -1)
            cv2.putText(frame, " | ".join(oraciones), (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            color_mano = (0, 200, 0) if mano_detectada else (0, 0, 200)
            cv2.circle(frame, (frame.shape[1] - 20, 20), 8, color_mano, -1)
            cv2.imshow("Prueba en vivo LSEC (Q para salir)", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()

    print("\nSi el reconocimiento se vio bien (pocas confusiones, pocos falsos "
          "positivos), seguí con: python 5_convertir_tflite.py")


if __name__ == "__main__":
    main()
