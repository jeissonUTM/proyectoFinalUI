"""
server_ws.py
============
Servidor WebSocket que corre el reconocimiento de señas en tiempo real
usando el modelo lsec_model.tflite. Flutter captura la cámara local y
envía cada frame JPEG como mensaje binario por WebSocket.
"""

import asyncio
import json
import os
import time
from http import HTTPStatus
import cv2
import numpy as np
from tflite_runtime.interpreter import Interpreter
import websockets
from importlib import import_module

import config

# Importar módulos de tu proyecto
captura_mod = import_module("1_capturar_keypoints")
preproc_mod = import_module("2_preprocesar_dataset")

ExtractorKeypoints = captura_mod.ExtractorKeypoints
VisionRunningMode = captura_mod.VisionRunningMode
normalizar_frame = preproc_mod.normalizar_frame
remuestrear_secuencia = preproc_mod.remuestrear_secuencia


class SegmentadorSenas:
    """Aísla una seña dentro de un flujo continuo de frames."""
    def __init__(self, min_frames=config.MIN_RAW_FRAMES, frames_espera=6):
        self.buffer = []
        self.frames_sin_mano = 0
        self.min_frames = min_frames
        self.frames_espera = frames_espera

    def procesar(self, vec: np.ndarray, mano_detectada: bool):
        if mano_detectada:
            self.buffer.append(vec)
            self.frames_sin_mano = 0
            return None

        if not self.buffer:
            return None

        self.frames_sin_mano += 1
        if self.frames_sin_mano < self.frames_espera:
            return None

        seq = np.array(self.buffer, dtype=np.float32)
        self.buffer = []
        self.frames_sin_mano = 0
        if len(seq) < self.min_frames:
            return None
        return seq


def preprocesar_para_modelo(seq_cruda: np.ndarray) -> np.ndarray:
    """Aplica exactamente los mismos pasos que en el entrenamiento."""
    seq_norm = np.stack([normalizar_frame(f) for f in seq_cruda])
    return remuestrear_secuencia(seq_norm, config.SEQUENCE_LENGTH)


def responder_http(connection, request):
    """Responde a las sondas HTTP de Render sin interferir con WebSocket."""
    if request.headers.get("Upgrade", "").lower() != "websocket":
        return connection.respond(HTTPStatus.OK, "Servicio WebSocket activo.\n")
    return None


async def manejar_cliente(websocket):
    print(f"Cliente conectado: {websocket.remote_address}")

    # Cargar modelo TFLite
    print("Cargando modelo TFLite...")
    interpreter = Interpreter(model_path=config.TFLITE_MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    segmentador = SegmentadorSenas()
    ultima_palabra = None
    ultima_vez = 0.0
    cooldown = 2.0
    umbral = 0.80

    try:
        with ExtractorKeypoints(running_mode=VisionRunningMode.VIDEO) as extractor:
            t0 = time.time()

            async for mensaje in websocket:
                if isinstance(mensaje, str):
                    try:
                        if json.loads(mensaje).get("type") == "stop":
                            break
                    except json.JSONDecodeError:
                        pass
                    continue

                frame = cv2.imdecode(
                    np.frombuffer(mensaje, dtype=np.uint8), cv2.IMREAD_COLOR
                )
                if frame is None:
                    continue
                timestamp_ms = int((time.time() - t0) * 1000)

                vec = extractor.process_bgr_frame(frame, timestamp_ms)

                porcion_manos = vec[config.POSE_FEATS:]
                mano_detectada = bool(np.any(porcion_manos != 0))

                resultado = segmentador.procesar(vec, mano_detectada)

                if resultado is not None:
                    entrada = preprocesar_para_modelo(resultado)
                    entrada = np.expand_dims(entrada, axis=0).astype(np.float32)

                    interpreter.set_tensor(input_details[0]['index'], entrada)
                    interpreter.invoke()
                    probs = interpreter.get_tensor(output_details[0]['index'])[0]

                    idx = int(np.argmax(probs))
                    confianza = float(probs[idx])
                    palabra = config.VOCABULARY[idx]

                    ahora = time.time()
                    es_repetida = (palabra == ultima_palabra) and (ahora - ultima_vez < cooldown)

                    if confianza >= umbral and not es_repetida:
                        print(f"Enviando: {palabra} ({confianza*100:.1f}%)")
                        await websocket.send(json.dumps({
                            "type": "prediction",
                            "palabra": palabra,
                            "confianza": confianza
                        }))
                        ultima_palabra = palabra
                        ultima_vez = ahora

    except websockets.exceptions.ConnectionClosed:
        print("Cliente desconectado")
    finally:
        print("Cliente finalizó la traducción.")


async def main():
    host = "0.0.0.0"
    port = int(os.environ.get("PORT", "8765"))
    async with websockets.serve(
        manejar_cliente, host, port, process_request=responder_http, max_size=2 * 1024 * 1024
    ):
        print(f"Servidor WebSocket iniciado en ws://{host}:{port}")
        print("Esperando conexión de Flutter... (envía video y predicciones)")
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
