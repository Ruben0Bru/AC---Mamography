"""Prototipo académico: clasificación de mamografías (benigno vs maligno).
Aprendizaje Computacional - Ingeniería de Sistemas.
*** NO diagnostica cáncer ni sirve para decisiones clínicas. ***"""
import os

import gradio as gr

import inferencia as inf

AVISO = "⚠️ Prototipo académico; no apto para diagnóstico ni decisiones clínicas"
SESION, ERROR_MODELO = inf.cargar_sesion()


def previsualizar(archivo):
    if archivo is None:
        return None, ""
    try:
        return inf.reducir(inf.a_uint8(inf.leer_imagen(archivo))), ""
    except inf.ImagenInvalida as e:
        return None, f"❌ **Archivo no válido.** {e}"


def clasificar(archivo, umbral, ver_gradcam):
    vacio = (None, None, None)
    if SESION is None:
        return (f"❌ **Modelo no disponible.** {ERROR_MODELO}", *vacio)
    if archivo is None:
        return ("ℹ️ Sube una imagen o un DICOM para comenzar.", *vacio)
    try:
        img_u8 = inf.a_uint8(inf.leer_imagen(archivo))
    except inf.ImagenInvalida as e:
        return (f"❌ **Archivo no válido.** {e}", *vacio)

    score, cam = inf.predecir(SESION, inf.preprocesar(img_u8))
    pantalla = inf.reducir(img_u8)

    if score >= umbral:
        veredicto = "🔴 **Clasificación del modelo: MALIGNO** (puntuación ≥ umbral)"
    else:
        veredicto = "🟢 **Clasificación del modelo: BENIGNO** (puntuación < umbral)"
    texto = (
        f"{veredicto}\n\n"
        f"- **Puntuación de malignidad:** `{score:.3f}`\n"
        f"- **Umbral de decisión (τ):** `{umbral:.2f}`\n\n"
        "> La puntuación **no es una probabilidad clínica calibrada** y la etiqueta corresponde a la "
        "patología de una lesión en el dataset de entrenamiento, **no** al diagnóstico de una persona.\n\n"
        f"**{AVISO}**"
    )
    cam_img = inf.superponer(pantalla, cam) if ver_gradcam else None
    return texto, {"Maligno": score, "Benigno": 1 - score}, pantalla, cam_img


CSS = """
#aviso {background:#b71c1c;color:white;padding:14px 18px;border-radius:10px;
        font-size:1.05rem;font-weight:700;text-align:center;}
#aviso p {color:white;margin:0;}
"""

with gr.Blocks(title="Clasificación de lesiones mamográficas — Prototipo académico", css=CSS) as demo:
    gr.Markdown(AVISO, elem_id="aviso")
    gr.Markdown(
        "# Clasificación de lesiones mamográficas (benigna / maligna)\n"
        "Proyecto final de **Aprendizaje Computacional** · ResNet18 con transferencia de aprendizaje "
        "entrenada sobre mamografías del dataset **CMMD** (TCIA)."
    )
    with gr.Row():
        with gr.Column():
            archivo = gr.File(label="Mamografía (DICOM, PNG, JPG, BMP o TIFF)",
                              file_types=inf.FORMATOS, type="filepath")
            umbral = gr.Slider(0.05, 0.95, value=inf.UMBRAL_DEFECTO, step=0.05,
                               label="Umbral de decisión (τ)",
                               info=f"Valor seleccionado en validación: {inf.UMBRAL_DEFECTO}. Bajarlo reduce "
                                    "falsos negativos pero aumenta falsos positivos.")
            ver_gradcam = gr.Checkbox(value=True, label="Mostrar Grad-CAM (zonas que más influyen)")
            btn = gr.Button("Clasificar", variant="primary")
            error_carga = gr.Markdown()
        with gr.Column():
            resultado = gr.Markdown("ℹ️ Sube una imagen o un DICOM para comenzar.")
            etiquetas = gr.Label(label="Puntuaciones del modelo (no calibradas)", num_top_classes=2)
    with gr.Row():
        img_pre = gr.Image(label="Imagen leída (escala de grises normalizada)", interactive=False)
        img_cam = gr.Image(label="Grad-CAM (clase Maligno)", interactive=False)

    with gr.Accordion("Alcance, formatos y limitaciones", open=False):
        gr.Markdown(
            "**Formatos admitidos:** DICOM (`.dcm`, 2D en escala de grises), PNG, JPG, BMP, TIFF. "
            f"Lado mínimo: {inf.LADO_MIN} px. Cualquier otro archivo se rechaza con un mensaje.\n\n"
            "**Preprocesamiento:** normalización min-max a 0–255 → escala de grises a 3 canales → "
            "redimensionado a 224×224 → normalización ImageNet (idéntico al entrenamiento).\n\n"
            "**Datos:** CMMD (Chinese Mammography Database, TCIA). Etiqueta *Benign/Malignant* de patología, "
            "asignada por mama (paciente + lateralidad). La unidad de análisis es la **imagen**.\n\n"
            "**Limitaciones:** un solo dataset de una población específica; sin validación externa; la "
            "puntuación no es una probabilidad calibrada; no sustituye la lectura de un radiólogo. Un "
            "resultado «benigno» **no** descarta cáncer y uno «maligno» **no** lo confirma.\n\n"
            "**Uso:** exclusivamente educativo."
        )
    gr.Markdown(f"<center><b>{AVISO}</b></center>")

    archivo.change(previsualizar, archivo, [img_pre, error_carga])
    btn.click(clasificar, [archivo, umbral, ver_gradcam], [resultado, etiquetas, img_pre, img_cam])

demo.queue(default_concurrency_limit=1, max_size=10)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)),
                ssr_mode=False, show_api=False)
