"""
Prototipo académico: clasificación de mamografías (benigno vs maligno).
Aprendizaje Computacional - Ingeniería de Sistemas.

*** NO diagnostica cáncer ni sirve para decisiones clínicas. ***
"""
import os

import gradio as gr
import matplotlib
import numpy as np
import pydicom
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

matplotlib.use("Agg")
from matplotlib import colormaps  # noqa: E402

# ----------------------------------------------------------------------------
# Configuración (debe coincidir con el notebook de entrenamiento)
# ----------------------------------------------------------------------------
RUTA_PESOS = os.environ.get("RUTA_PESOS", "resnet18_linea_base.pth")
UMBRAL_DEFECTO = 0.30          # τ elegido en VALIDACIÓN (Hipótesis 2)
TAM_ENTRADA = 224
TAM_MIN_LADO = 64              # lado mínimo aceptado (px)
FORMATOS = [".dcm", ".dicom", ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]
AVISO = "⚠️ Prototipo académico; no apto para diagnóstico ni decisiones clínicas"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Mismo pipeline que `transform_base` del notebook
TRANSFORM = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((TAM_ENTRADA, TAM_ENTRADA)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# ----------------------------------------------------------------------------
# Modelo
# ----------------------------------------------------------------------------
def cargar_modelo():
    modelo = models.resnet18(weights=None)
    modelo.fc = nn.Linear(modelo.fc.in_features, 2)
    if not os.path.exists(RUTA_PESOS):
        return None, f"No se encontró el archivo de pesos «{RUTA_PESOS}»."
    try:
        estado = torch.load(RUTA_PESOS, map_location=DEVICE)
        modelo.load_state_dict(estado)
    except Exception as e:  # noqa: BLE001
        return None, f"No se pudieron cargar los pesos: {e}"
    return modelo.to(DEVICE).eval(), None


MODELO, ERROR_MODELO = cargar_modelo()


# ----------------------------------------------------------------------------
# Lectura y preprocesamiento
# ----------------------------------------------------------------------------
class ImagenInvalida(Exception):
    pass


def leer_imagen(ruta: str) -> np.ndarray:
    """Devuelve una matriz 2D float32 (escala de grises original)."""
    ext = os.path.splitext(ruta)[1].lower()
    if ext not in FORMATOS:
        raise ImagenInvalida(
            f"Formato «{ext or 'desconocido'}» no admitido. "
            f"Formatos válidos: {', '.join(FORMATOS)}."
        )
    try:
        if ext in (".dcm", ".dicom"):
            ds = pydicom.dcmread(ruta)
            arr = ds.pixel_array
            if arr.ndim != 2:
                raise ImagenInvalida(
                    "El DICOM no es una imagen 2D en escala de grises "
                    "(¿multiframe o a color?)."
                )
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
    if min(arr.shape) < TAM_MIN_LADO:
        raise ImagenInvalida(
            f"La imagen es demasiado pequeña ({arr.shape[1]}x{arr.shape[0]} px). "
            f"Mínimo {TAM_MIN_LADO}x{TAM_MIN_LADO}."
        )
    if not np.isfinite(arr).all():
        raise ImagenInvalida("La imagen contiene valores no numéricos.")
    return arr


def a_uint8(arr: np.ndarray) -> np.ndarray:
    """Normalización min-max a [0,255], idéntica al notebook."""
    mn, mx = float(arr.min()), float(arr.max())
    if mx <= mn:
        raise ImagenInvalida("La imagen es uniforme (sin contenido).")
    return ((arr - mn) / (mx - mn) * 255.0).astype(np.uint8)


# ----------------------------------------------------------------------------
# Inferencia + Grad-CAM (opcional)
# ----------------------------------------------------------------------------
def puntuacion_malignidad(x: torch.Tensor) -> float:
    with torch.no_grad():
        return torch.softmax(MODELO(x), dim=1)[0, 1].item()


def grad_cam(x: torch.Tensor) -> np.ndarray:
    """Mapa Grad-CAM (0-1, 7x7) de la clase 'Maligno' sobre layer4."""
    activ, grads = {}, {}

    def fwd_hook(_m, _i, out):
        activ["a"] = out
        out.register_hook(lambda g: grads.update(g=g))

    h = MODELO.layer4.register_forward_hook(fwd_hook)
    try:
        MODELO.zero_grad()
        with torch.enable_grad():
            salida = MODELO(x)
            salida[0, 1].backward()
        pesos = grads["g"].mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((pesos * activ["a"]).sum(dim=1))[0].detach().cpu().numpy()
    finally:
        h.remove()
    cam -= cam.min()
    return cam / cam.max() if cam.max() > 0 else cam


def superponer(img_u8: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    alto, ancho = img_u8.shape
    cam_img = Image.fromarray((cam * 255).astype(np.uint8)).resize((ancho, alto), Image.BILINEAR)
    mapa = colormaps["jet"](np.array(cam_img) / 255.0)[..., :3]
    base = np.stack([img_u8] * 3, axis=-1) / 255.0
    mezcla = np.clip((1 - alpha) * base + alpha * mapa, 0, 1)
    return (mezcla * 255).astype(np.uint8)


# ----------------------------------------------------------------------------
# Lógica de la interfaz
# ----------------------------------------------------------------------------
def previsualizar(archivo):
    """Muestra la imagen apenas se carga (o limpia si hay error)."""
    if archivo is None:
        return None, ""
    try:
        return a_uint8(leer_imagen(archivo)), ""
    except ImagenInvalida as e:
        return None, f"❌ **Archivo no válido.** {e}"


def clasificar(archivo, umbral, ver_gradcam):
    vacio = (None, None, None)
    if MODELO is None:
        return (f"❌ **Modelo no disponible.** {ERROR_MODELO}", *vacio)
    if archivo is None:
        return ("ℹ️ Sube una imagen o un DICOM para comenzar.", *vacio)

    try:
        arr = leer_imagen(archivo)
        img_u8 = a_uint8(arr)
    except ImagenInvalida as e:
        return (f"❌ **Archivo no válido.** {e}", *vacio)

    x = TRANSFORM(img_u8).unsqueeze(0).to(DEVICE)
    score = puntuacion_malignidad(x)
    es_sospechosa = score >= umbral

    if es_sospechosa:
        veredicto = "🔴 **Clasificación del modelo: MALIGNO** (puntuación ≥ umbral)"
    else:
        veredicto = "🟢 **Clasificación del modelo: BENIGNO** (puntuación < umbral)"

    texto = (
        f"{veredicto}\n\n"
        f"- **Puntuación de malignidad:** `{score:.3f}`\n"
        f"- **Umbral de decisión (τ):** `{umbral:.2f}`\n\n"
        "> La puntuación **no es una probabilidad clínica calibrada** y la etiqueta "
        "corresponde a la patología de una lesión en el dataset de entrenamiento, "
        "**no** al diagnóstico de una persona.\n\n"
        f"**{AVISO}**"
    )
    puntuaciones = {"Maligno": float(score), "Benigno": float(1 - score)}

    cam_img = None
    if ver_gradcam:
        try:
            cam_img = superponer(img_u8, grad_cam(x))
        except Exception:  # noqa: BLE001
            cam_img = None

    return texto, puntuaciones, img_u8, cam_img


# ----------------------------------------------------------------------------
# Interfaz Gradio
# ----------------------------------------------------------------------------
CSS = """
#aviso {background:#b71c1c;color:white;padding:14px 18px;border-radius:10px;
        font-size:1.05rem;font-weight:700;text-align:center;}
#aviso p {color:white;margin:0;}
"""

with gr.Blocks(title="Clasificación de lesiones mamográficas — Prototipo académico", css=CSS) as demo:
    gr.Markdown(f"{AVISO}", elem_id="aviso")
    gr.Markdown(
        "# Clasificación de lesiones mamográficas (benigna / maligna)\n"
        "Proyecto final de **Aprendizaje Computacional** · ResNet18 con transferencia de "
        "aprendizaje entrenada sobre mamografías del dataset **CMMD** (TCIA)."
    )

    with gr.Row():
        with gr.Column(scale=1):
            archivo = gr.File(
                label="Mamografía (DICOM, PNG, JPG, BMP o TIFF)",
                file_types=FORMATOS,
                type="filepath",
            )
            umbral = gr.Slider(
                0.05, 0.95, value=UMBRAL_DEFECTO, step=0.05,
                label="Umbral de decisión (τ)",
                info=f"Valor seleccionado en validación: {UMBRAL_DEFECTO}. "
                     "Bajarlo reduce falsos negativos pero aumenta falsos positivos.",
            )
            ver_gradcam = gr.Checkbox(value=True, label="Mostrar Grad-CAM (zonas que más influyen)")
            btn = gr.Button("Clasificar", variant="primary")
            error_carga = gr.Markdown()

        with gr.Column(scale=1):
            resultado = gr.Markdown("ℹ️ Sube una imagen o un DICOM para comenzar.")
            etiquetas = gr.Label(label="Puntuaciones del modelo (no calibradas)", num_top_classes=2)

    with gr.Row():
        img_pre = gr.Image(label="Imagen leída (escala de grises normalizada)", interactive=False)
        img_cam = gr.Image(label="Grad-CAM (clase Maligno)", interactive=False)

    with gr.Accordion("Alcance, formatos y limitaciones", open=False):
        gr.Markdown(
            "**Formatos admitidos:** DICOM (`.dcm`, 2D en escala de grises), PNG, JPG, BMP, TIFF. "
            f"Lado mínimo: {TAM_MIN_LADO} px. Cualquier otro archivo se rechaza con un mensaje.\n\n"
            "**Preprocesamiento:** normalización min-max a 0–255 → escala de grises a 3 canales → "
            "redimensionado a 224×224 → normalización ImageNet (idéntico al entrenamiento).\n\n"
            "**Datos:** CMMD (Chinese Mammography Database, TCIA). Etiqueta *Benign/Malignant* "
            "proveniente de patología, asignada por mama (paciente + lateralidad). "
            "La unidad de análisis es la **imagen**.\n\n"
            "**Limitaciones:** un solo dataset de una población específica; sin validación externa; "
            "la puntuación no es una probabilidad calibrada; no sustituye la lectura de un radiólogo. "
            "Un resultado «benigno» **no** descarta cáncer y uno «maligno» **no** lo confirma.\n\n"
            "**Uso:** exclusivamente educativo."
        )

    gr.Markdown(f"<center><b>{AVISO}</b></center>")

    archivo.change(previsualizar, archivo, [img_pre, error_carga])
    btn.click(clasificar, [archivo, umbral, ver_gradcam], [resultado, etiquetas, img_pre, img_cam])

if __name__ == "__main__":
    demo.launch()
