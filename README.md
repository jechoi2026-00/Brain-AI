# Brain-AI

MRI-based Brain Tumor Prediction and Visualization System

## Overview

**Brain-AI** is a research-oriented deep learning project for **brain tumor detection, segmentation, and visualization using MRI images**.  
The project combines **YOLOv8 segmentation**, **3D U-Net-based volumetric analysis**, and a **Streamlit web interface** to provide an interactive demonstration of AI-assisted medical imaging analysis.

This repository was developed as an academic project and is intended for **research and educational purposes only**.

## Key Features

- Brain tumor detection from MRI slices using **YOLOv8-Segmentation**
- 3D-oriented analysis concept based on **3D U-Net / volumetric medical imaging workflow**
- Interactive **Streamlit dashboard** for image upload and inference
- Visualization of:
  - AI-segmented tumor regions
  - Detection confidence
  - Anatomical 3D localization
  - Diagnostic interpretation panel
- Plotly-based interactive 3D rendering
- Medical AI demonstration workflow for educational presentation

## Streamlit App Preview

Below are example outputs from the Streamlit application.

### 1) Brain Tumor 3D Diagnosis XAI 
<img width="1208" height="1106" alt="그림01" src="https://github.com/user-attachments/assets/a0072ff8-394d-4a06-b3bf-1b7479e44ea6" />

### 2) Precision Diagnostic Interpretation

<img width="1600" height="1123" alt="그림03" src="https://github.com/user-attachments/assets/4f00091c-7861-43b4-8c57-5e5bf2d55999" />


## Project Background

Brain tumor diagnosis requires high precision and often depends on the integrated interpretation of radiology, pathology, and clinical findings. This project explores how AI-based segmentation and visualization can support:

- clearer understanding of tumor structure,
- improved communication of imaging findings,
- educational demonstration of AI-assisted diagnosis,
- and future extension toward medical decision support systems.

## Models and Methods

### 1. YOLOv8s
Used for 2D real-time object detection and segmentation on MRI slices.

Reported project presentation points include:
- Epochs: 5
- Image size: 240
- Batch size: 16
- Segmentation-based tumor detection
- Performance indicators such as precision, recall, and mAP

### 2. 3D U-Net
Used as the volumetric segmentation concept for 3D MRI analysis.

Key workflow:
- 3D medical image preprocessing
- MONAI transforms
- volumetric segmentation learning
- Dice-loss-based optimization

### 3. Explainable / Interpretable Visualization
The app and presentation also emphasize visual interpretation of inference results, including confidence reporting and anatomical localization.

## Dataset

This project is based on a **brain tumor MRI dataset from Kaggle / BraTS-related sources**.

Presentation summary:
- Adult glioma MRI cases
- 1,251 cases
- 4 MRI modalities per case
- segmentation targets including ET, TC, and WT
- 3D-to-2D conversion for slice-based training workflow

> Please make sure that your actual public repository clearly names the dataset version you used (for example, Kaggle BraTS-derived data) and follows the dataset's usage terms.

## Repository Structure

```bash
Brain-AI/
├── app.py
├── README.md
├── LICENSE
├── requirements.txt
├── packages.txt
├── .gitignore
├── assets/
│   └── screenshots/
│       ├── streamlit_preview_1.png
│       └── streamlit_preview_2.png
├── notebooks/
│   └── GBM_training_clean.ipynb   # recommended cleaned notebook
└── models/
    └── best.pt                    # not recommended for direct Git tracking
```

## Main Application

The Streamlit application (`app.py`) includes:

- model loading with `ultralytics.YOLO`
- MRI image upload
- tumor segmentation inference
- 2D result visualization
- 3D anatomical localization using Plotly
- confidence and lesion summary metrics
- diagnostic interpretation blocks

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/jechoi2026-00/Brain-AI.git
cd Brain-AI
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

If needed for Streamlit Cloud or Linux-based deployment, system packages can be listed in `packages.txt`.

### 3. Prepare model weights

Place your trained YOLO model file in the project root:

```bash
best.pt
```

The current app expects:

```python
model = YOLO('best.pt')
```

### 4. Run the Streamlit app

```bash
streamlit run app.py
```

## Requirements

Typical Python dependencies include:

- streamlit
- opencv-python-headless
- numpy
- torch
- ultralytics
- pillow
- plotly
- pandas

## Recommended Cleanup Before Publishing

Before making the repository fully public, it is strongly recommended to:

- remove **Kaggle API tokens** or any credential files,
- remove unnecessary notebook outputs,
- avoid uploading large model weights directly to GitHub,
- keep only cleaned and reproducible training notebooks,
- add screenshots and project assets in organized folders,
- verify that no externally sourced material is incorrectly claimed as your copyright.

## Limitations

- This project is an academic prototype, not a clinical product.
- The current implementation is not a certified medical device.
- Model performance may vary depending on preprocessing, dataset split, and training scale.
- Public deployment should not imply real-world medical diagnosis capability.

## Future Work

Possible extensions include:

- multimodal MRI fusion
- domain-specific data augmentation
- better explainable AI methods such as Grad-CAM integration
- federated learning across institutions
- extension to stroke, Alzheimer's disease, or other brain disorders
- richer 3D web dashboard interfaces

## License

This project is licensed under the **MIT License**.
See the [LICENSE](LICENSE) file for details.

## Copyright Notice

Copyright (c) 2026 Jeongeun Choi

The copyright and MIT license in this repository apply **only to the original code and original documentation created by the repository author**.

They **do not automatically apply** to external materials, including but not limited to:

- datasets obtained from Kaggle or BraTS-related sources,
- pretrained model weights,
- third-party libraries,
- externally sourced figures, images, or media,
- papers, challenge materials, or other content owned by their respective authors.

All external materials remain the property of their original copyright holders and must be used according to their own licenses and terms.

