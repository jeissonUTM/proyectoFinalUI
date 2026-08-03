"""
server_ws.py
============
Servidor WebSocket que corre el reconocimiento de señas en tiempo real
usando el modelo lsec_model.tflite, envía la palabra traducida y también
los frames de video codificados en base64 para mostrarlos en Flutter.

CORRECCIÓN (parpadeo + cámara que no se apaga):
El bucle principal antes NUNCA escuchaba mensajes entrantes del cliente
(solo hacía `send`), así que el servidor no se enteraba de que Flutter
quería detener la traducción hasta que un `send` fallaba por conexión
cerrada -> la cámara seguía prendida un rato de más.

Ahora se lanza una tarea en paralelo (`escuchar_stop`) que queda
esperando mensajes del cliente. En cuanto Flutter manda {"type":"stop"}
o cierra el socket, se activa un `asyncio.Event` y el bucle principal
lo revisa en cada iteración para cortar y liberar la cámara al toque.
"""

import asyncio
import json
import os
import time
import base64
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


async def escuchar_stop(websocket, evento_stop: asyncio.Event):
    """Corre en paralelo al bucle de cámara. Se queda esperando mensajes
    entrantes del cliente (Flutter). En cuanto llega {"type":"stop"} o
    el socket se cierra desde el otro lado, activa evento_stop para que
    el bucle principal corte y libere la cámara de inmediato."""
    try:
        async for mensaje in websocket:
            try:
                data = json.loads(mensaje)
            except (json.JSONDecodeError, TypeError):
                continue
            if data.get("type") == "stop":
                print("Stop recibido desde Flutter, liberando cámara...")
                evento_stop.set()
                break
    except websockets.exceptions.ConnectionClosed:
        pass
    finally:
        # Si el socket se cerró sin mandar "stop" explícito, igual
        # cortamos el bucle principal.
        evento_stop.set()


async def manejar_cliente(websocket):
    print(f"Cliente conectado: {websocket.remote_address}")

    evento_stop = asyncio.Event()
    tarea_escucha = asyncio.create_task(escuchar_stop(websocket, evento_stop))

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

    frame_counter = 0
    FRAME_INTERVAL = 2  # enviar un frame cada 2 iteraciones (~15 FPS si la cámara va a 30)

    cap = None
    try:
        with ExtractorKeypoints(running_mode=VisionRunningMode.VIDEO) as extractor:
            cap = cv2.VideoCapture(0)
            t0 = time.time()

            while not evento_stop.is_set():
                ok, frame = cap.read()
                if not ok:
                    break

                frame = cv2.flip(frame, 1)
                timestamp_ms = int((time.time() - t0) * 1000)

                vec = extractor.process_bgr_frame(frame, timestamp_ms)

                porcion_manos = vec[config.POSE_FEATS:]
                mano_detectada = bool(np.any(porcion_manos != 0))

                resultado = segmentador.procesar(vec, mano_detectada)

                frame_counter += 1
                if frame_counter % FRAME_INTERVAL == 0:
                    _, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                    jpeg_b64 = base64.b64encode(jpeg.tobytes()).decode('utf-8')
                    await websocket.send(json.dumps({
                        "type": "frame",
                        "data": jpeg_b64
                    }))

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

                # Cede el control al event loop en cada vuelta. Esto es
                # lo que permite que la tarea `escuchar_stop` (y por lo
                # tanto la detección del cierre/stop) se procese casi al
                # instante, en vez de recién en el próximo `send`.
                await asyncio.sleep(0)

    except websockets.exceptions.ConnectionClosed:
        print("Cliente desconectado")
    finally:
        if cap is not None:
            cap.release()
        tarea_escucha.cancel()
        print("Cámara liberada.")


async def main():
    host = "0.0.0.0"
    port = int(os.environ.get("PORT", "8765"))
    async with websockets.serve(manejar_cliente, host, port):
        print(f"Servidor WebSocket iniciado en ws://{host}:{port}")
        print("Esperando conexión de Flutter... (envía video y predicciones)")
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
