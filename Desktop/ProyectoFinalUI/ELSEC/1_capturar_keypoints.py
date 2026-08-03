"""
1_capturar_keypoints.py
========================
Extrae keypoints de manos y pose usando la API MODERNA de MediaPipe Tasks
(PoseLandmarker + HandLandmarker).

NOTA TÉCNICA IMPORTANTE (verificado empíricamente en julio 2026):
La antigua API `mp.solutions.holistic` fue retirada de las versiones
recientes del paquete `mediapipe` (a partir de ~0.10.x ya no existe
`mediapipe.solutions`; produce `AttributeError`). Este script usa en su
lugar `mediapipe.tasks.vision.PoseLandmarker` y
`mediapipe.tasks.vision.HandLandmarker`, que son las tareas estables y
documentadas oficialmente. (MediaPipe también expone ya una clase
`HolisticLandmarker` combinada, pero Google la marca como "early preview"
y hay reportes de errores de runtime y URLs de modelo inconsistentes;
por eso se prefiere aquí la combinación Pose+Hand, más madura).

No usamos FaceLandmarker: para un vocabulario de 50 palabras aisladas
(no oraciones con gramática no-manual) las manos y el torso concentran
casi toda la información discriminativa, y omitir la cara reduce el
tamaño del dataset necesario y acelera la extracción. Ver README.md
sección 3 si luego se desea añadir un subconjunto de landmarks faciales.

USO
---
1) Descargar los modelos .task (una sola vez, requiere internet):

   mkdir -p mp_models
   curl -L -o mp_models/pose_landmarker_lite.task \\
       https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task
   curl -L -o mp_models/hand_landmarker.task \\
       https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task

2a) Modo lote (recomendado): organiza tus videos en
    dataset/raw_videos/<PALABRA>/*.mp4  y luego ejecuta:

    python 1_capturar_keypoints.py --modo video

2b) Modo webcam interactivo (para grabar tus propias muestras):

    python 1_capturar_keypoints.py --modo webcam --palabra HOLA
    (ESPACIO = iniciar/detener grabación de una repetición, ESC = salir)
"""

import argparse
import glob
import os
import time

import cv2
import numpy as np
import mediapipe as mp

import config

BaseOptions = mp.tasks.BaseOptions
VisionRunningMode = mp.tasks.vision.RunningMode


class ExtractorKeypoints:
    """Envuelve PoseLandmarker + HandLandmarker y produce un vector de
    178 features (52 pose + 63 mano izq + 63 mano der) por frame."""

    def __init__(self, running_mode=VisionRunningMode.VIDEO):
        if not os.path.exists(config.POSE_MODEL_PATH):
            raise FileNotFoundError(
                f"No se encontró {config.POSE_MODEL_PATH}. "
                "Descarga los modelos .task (ver docstring de este archivo)."
            )
        if not os.path.exists(config.HAND_MODEL_PATH):
            raise FileNotFoundError(
                f"No se encontró {config.HAND_MODEL_PATH}. "
                "Descarga los modelos .task (ver docstring de este archivo)."
            )

        pose_options = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=config.POSE_MODEL_PATH),
            running_mode=running_mode,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        hand_options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=config.HAND_MODEL_PATH),
            running_mode=running_mode,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.pose_landmarker = mp.tasks.vision.PoseLandmarker.create_from_options(pose_options)
        self.hand_landmarker = mp.tasks.vision.HandLandmarker.create_from_options(hand_options)
        self._running_mode = running_mode

    def close(self):
        self.pose_landmarker.close()
        self.hand_landmarker.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    @staticmethod
    def _empty_pose():
        return np.zeros(config.POSE_FEATS, dtype=np.float32)

    @staticmethod
    def _empty_hand():
        return np.zeros(config.HAND_FEATS, dtype=np.float32)

    def _pose_to_vec(self, pose_result) -> np.ndarray:
        if not pose_result.pose_landmarks:
            return self._empty_pose()
        lms = pose_result.pose_landmarks[0]  # una sola persona (num_poses=1)
        vec = np.zeros((config.N_POSE, 4), dtype=np.float32)
        for i, idx in enumerate(config.POSE_LANDMARKS_IDX):
            lm = lms[idx]
            vec[i] = [lm.x, lm.y, lm.z, lm.visibility]
        return vec.flatten()

    def _hands_to_vecs(self, hand_result):
        left = self._empty_hand()
        right = self._empty_hand()
        for lms, handed in zip(hand_result.hand_landmarks, hand_result.handedness):
            # NOTA: MediaPipe clasifica "Left"/"Right" asumiendo la imagen
            # como espejo (vista selfie). Si capturas con cámara frontal
            # (la más común para grabarse uno mismo señando), voltea el
            # frame horizontalmente (cv2.flip(frame, 1)) ANTES de pasarlo
            # aquí para que "Left"/"Right" correspondan a la mano real del
            # señante. Este script ya hace ese flip en modo webcam.
            label = handed[0].category_name  # "Left" o "Right"
            arr = np.array([[lm.x, lm.y, lm.z] for lm in lms], dtype=np.float32).flatten()
            if label == "Left":
                left = arr
            else:
                right = arr
        return left, right

    def process_bgr_frame(self, frame_bgr: np.ndarray, timestamp_ms: int) -> np.ndarray:
        """Procesa un frame BGR (formato OpenCV) y devuelve un vector (178,)."""
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        if self._running_mode == VisionRunningMode.VIDEO:
            pose_result = self.pose_landmarker.detect_for_video(mp_image, timestamp_ms)
            hand_result = self.hand_landmarker.detect_for_video(mp_image, timestamp_ms)
        else:
            pose_result = self.pose_landmarker.detect(mp_image)
            hand_result = self.hand_landmarker.detect(mp_image)

        pose_vec = self._pose_to_vec(pose_result)
        left_vec, right_vec = self._hands_to_vecs(hand_result)
        return np.concatenate([pose_vec, left_vec, right_vec])  # (178,)


def procesar_video(path_video: str, extractor: ExtractorKeypoints) -> np.ndarray:
    """Devuelve un array (T, 178) con un vector de keypoints por frame del video."""
    cap = cv2.VideoCapture(path_video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames_out = []
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        timestamp_ms = int((frame_idx / fps) * 1000)
        vec = extractor.process_bgr_frame(frame, timestamp_ms)
        frames_out.append(vec)
        frame_idx += 1
    cap.release()
    return np.array(frames_out, dtype=np.float32)


def modo_video():
    """Recorre dataset/raw_videos/<PALABRA>/*.mp4 y guarda un .npy por video
    en dataset/keypoints/<PALABRA>/."""
    with ExtractorKeypoints(running_mode=VisionRunningMode.VIDEO) as extractor:
        for palabra in config.VOCABULARY:
            carpeta_in = os.path.join(config.RAW_VIDEOS_DIR, palabra)
            carpeta_out = os.path.join(config.KEYPOINTS_DIR, palabra)
            os.makedirs(carpeta_out, exist_ok=True)
            videos = sorted(glob.glob(os.path.join(carpeta_in, "*.mp4")))
            if not videos:
                print(f"[AVISO] Sin videos para '{palabra}' en {carpeta_in}")
                continue
            for i, video_path in enumerate(videos):
                seq = procesar_video(video_path, extractor)
                if seq.shape[0] < config.MIN_RAW_FRAMES:
                    print(f"  [OMITIDO] {video_path}: solo {seq.shape[0]} frames")
                    continue
                out_path = os.path.join(carpeta_out, f"{palabra}_{i:03d}.npy")
                np.save(out_path, seq)
                print(f"  [OK] {out_path}  shape={seq.shape}")

""" el int =0 es la camara : 0 camara de la laptop o 1 es la camara secundaria """
def modo_webcam(palabra: str, camara_id: int = 0):
    """Captura interactiva por webcam para una palabra específica.
    ESPACIO: iniciar/detener grabación de una repetición.  ESC: salir."""
    carpeta_out = os.path.join(config.KEYPOINTS_DIR, palabra)
    os.makedirs(carpeta_out, exist_ok=True)
    existentes = len(glob.glob(os.path.join(carpeta_out, "*.npy")))

    cap = cv2.VideoCapture(1)
    grabando = False
    buffer_frames = []
    contador = existentes

    with ExtractorKeypoints(running_mode=VisionRunningMode.VIDEO) as extractor:
        t0 = time.time()
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)  # espejo: vista selfie natural + handedness correcto
            timestamp_ms = int((time.time() - t0) * 1000)
            vec = extractor.process_bgr_frame(frame, timestamp_ms)

            if grabando:
                buffer_frames.append(vec)
                cv2.circle(frame, (30, 30), 12, (0, 0, 255), -1)
                cv2.putText(frame, f"REC {len(buffer_frames)}", (50, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.putText(frame, f"Palabra: {palabra}  |  Muestras guardadas: {contador}",
                        (10, frame.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 255, 255), 2)
            cv2.imshow("Captura LSEC - ESPACIO=grabar  ESC=salir", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
            elif key == 32:  # ESPACIO
                if not grabando:
                    grabando = True
                    buffer_frames = []
                else:
                    grabando = False
                    seq = np.array(buffer_frames, dtype=np.float32)
                    if seq.shape[0] >= config.MIN_RAW_FRAMES:
                        out_path = os.path.join(carpeta_out, f"{palabra}_{contador:03d}.npy")
                        np.save(out_path, seq)
                        contador += 1
                        print(f"[OK] Guardado {out_path}  shape={seq.shape}")
                    else:
                        print(f"[DESCARTADO] Muy corto ({seq.shape[0]} frames)")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Captura de keypoints LSEC")
    parser.add_argument("--modo", choices=["video", "webcam"], default="video")
    parser.add_argument("--palabra", type=str, default=None,
                         help="Palabra a grabar en modo webcam (debe estar en config.VOCABULARY)")
    parser.add_argument("--camara", type=int, default=0)
    args = parser.parse_args()

    if args.modo == "video":
        modo_video()
    else:
        if args.palabra is None or args.palabra not in config.VOCABULARY:
            raise ValueError(f"--palabra debe ser una de: {config.VOCABULARY}")
        modo_webcam(args.palabra, args.camara)
