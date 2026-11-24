---
title: Clasificación de Mamografías (Prototipo Académico)
emoji: 🩻
colorFrom: pink
colorTo: red
sdk: gradio
sdk_version: 5.49.1
app_file: app.py
pinned: false
---

# Clasificación de lesiones mamográficas benignas/malignas — Prototipo académico

> ⚠️ **Prototipo académico; no apto para diagnóstico ni decisiones clínicas.**

Interfaz Gradio del proyecto final de Aprendizaje Computacional. Modelo: ResNet18 preentrenada
(ImageNet) con transferencia de aprendizaje, entrenada sobre el dataset CMMD (TCIA).

## Ejecutar localmente
```bash
pip install -r requirements.txt
pip install gradio==5.49.1
# coloca resnet18_linea_base.pth en esta carpeta
python app.py
```
Abre http://127.0.0.1:7860.

## Archivos
- `app.py` – interfaz y lógica de inferencia (+ Grad-CAM).
- `resnet18_linea_base.pth` – pesos entrenados en el notebook (descargar de Colab).
- `requirements.txt` – dependencias.

## Reproducibilidad
Semilla 42 · división por paciente 70/15/15 (`GroupShuffleSplit`) · ResNet18 · Adam lr=1e-4 ·
10 épocas · batch 32 · umbral τ=0.30 elegido en validación. No se publican datos del dataset.
