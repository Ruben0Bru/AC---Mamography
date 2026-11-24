# Clasificación de lesiones mamográficas benignas/malignas — Prototipo académico

> ⚠️ **Prototipo académico; no apto para diagnóstico ni decisiones clínicas.**

Interfaz Gradio del proyecto final de Aprendizaje Computacional (Universidad de Córdoba).
Modelo: ResNet18 con transferencia de aprendizaje, entrenado con el dataset CMMD (TCIA).

## Ejecutar localmente
```bash
pip install -r requirements.txt
python app.py        # http://127.0.0.1:7860
```
Requiere `resnet18_linea_base.onnx` en esta carpeta (se genera con `exportar_onnx.py` desde el `.pth` entrenado).

## Archivos
- `app.py` – interfaz Gradio.
- `inferencia.py` – lectura, preprocesamiento, inferencia y Grad-CAM (onnxruntime, sin PyTorch).
- `exportar_onnx.py` – convierte el `.pth` a ONNX y verifica equivalencia (se ejecuta en Colab).
- `resnet18_linea_base.onnx` – modelo exportado.
- `requirements.txt` – dependencias.

## Despliegue en Render
Build: `pip install -r requirements.txt` · Start: `python app.py` · variable `PYTHON_VERSION=3.11.9`.
