"""Lógica de lectura, preprocesamiento e inferencia (sin PyTorch ni Gradio).
Replica el pipeline del notebook: min-max a 0-255 -> gris a 3 canales ->
resize 224x224 (bilineal) -> normalización ImageNet."""
import os

import numpy as np
import onnxruntime as ort
from PIL import Image

RUTA_MODELO = os.environ.get("MODEL_PATH", "resnet18_linea_base.onnx")
UMBRAL_DEFECTO = 0.30
TAM = 224
LADO_MIN = 64
FORMATOS = [".dcm", ".dicom", ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]
MEDIA = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)


class ImagenInvalida(Exception):
    pass


def cargar_sesion():
    if not os.path.exists(RUTA_MODELO):
        return None, f"No se encontró el archivo del modelo «{RUTA_MODELO}»."
    try:
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 2
        return ort.InferenceSession(RUTA_MODELO, opts, providers=["CPUExecutionProvider"]), None
    except Exception as e:  # noqa: BLE001
        return None, f"No se pudo cargar el modelo: {e}"


def leer_imagen(ruta: str) -> np.ndarray:
    ext = os.path.splitext(ruta)[1].lower()
    if ext not in FORMATOS:
        raise ImagenInvalida(
            f"Formato «{ext or 'desconocido'}» no admitido. Formatos válidos: {', '.join(FORMATOS)}.")
    try:
        if ext in (".dcm", ".dicom"):
            import pydicom
            arr = pydicom.dcmread(ruta).pixel_array
            if arr.ndim != 2:
                raise ImagenInvalida("El DICOM no es una imagen 2D en escala de grises (¿multiframe o a color?).")
        else:
            img = Image.open(ruta)
            if img.mode not in ("I;16", "I", "F"):
                img = img.convert("L")
            arr = np.array(img)
            if arr.ndim != 2:
                raise ImagenInvalida("No se pudo interpretar la imagen como escala de grises.")
    except ImagenInvalida:
        raise
    except Exception as e:  # noqa: BLE001
        raise ImagenInvalida(f"No se pudo leer el archivo como imagen válida ({type(e).__name__}).") from e

    arr = arr.astype(np.float32)
    if min(arr.shape) < LADO_MIN:
        raise ImagenInvalida(f"Imagen demasiado pequeña ({arr.shape[1]}x{arr.shape[0]} px). Mínimo {LADO_MIN}x{LADO_MIN}.")
    if not np.isfinite(arr).all():
        raise ImagenInvalida("La imagen contiene valores no numéricos.")
    return arr


def a_uint8(arr: np.ndarray) -> np.ndarray:
    mn, mx = float(arr.min()), float(arr.max())
    if mx <= mn:
        raise ImagenInvalida("La imagen es uniforme (sin contenido).")
    return ((arr - mn) / (mx - mn) * 255.0).astype(np.uint8)


def reducir(img_u8: np.ndarray, max_lado: int = 900) -> np.ndarray:
    """Versión más liviana solo para mostrar en pantalla."""
    h, w = img_u8.shape
    if max(h, w) <= max_lado:
        return img_u8
    f = max_lado / max(h, w)
    return np.asarray(Image.fromarray(img_u8).resize((int(w * f), int(h * f)), Image.Resampling.BILINEAR))


def preprocesar(img_u8: np.ndarray) -> np.ndarray:
    pil = Image.fromarray(img_u8).resize((TAM, TAM), Image.Resampling.BILINEAR)
    a = np.asarray(pil, dtype=np.float32) / 255.0
    a = np.repeat(a[None], 3, axis=0)
    return ((a - MEDIA) / STD)[None].astype(np.float32)


def predecir(sesion, x: np.ndarray):
    """Devuelve (puntuación de malignidad, mapa CAM 7x7 normalizado a 0-1)."""
    logits, cam = sesion.run(None, {"input": x})
    z = logits[0] - logits[0].max()
    score = float(np.exp(z[1]) / np.exp(z).sum())
    cam = cam[0] - cam[0].min()
    cam = cam / cam.max() if cam.max() > 0 else cam
    return score, cam


def _jet(v: np.ndarray) -> np.ndarray:
    r = np.clip(1.5 - np.abs(4 * v - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * v - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * v - 1), 0, 1)
    return np.stack([r, g, b], axis=-1)


def superponer(img_u8: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    h, w = img_u8.shape
    cam_u8 = Image.fromarray((cam * 255).astype(np.uint8)).resize((w, h), Image.Resampling.BILINEAR)
    mapa = _jet(np.asarray(cam_u8, dtype=np.float32) / 255.0)
    base = np.repeat(img_u8[..., None], 3, axis=-1) / 255.0
    return (np.clip((1 - alpha) * base + alpha * mapa, 0, 1) * 255).astype(np.uint8)
