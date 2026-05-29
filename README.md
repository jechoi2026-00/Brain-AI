# Brain-AI: MRI-Based Brain Tumor Segmentation and 3D Visualization

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-red)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-orange)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Segmentation-green)
![MONAI](https://img.shields.io/badge/MONAI-3D%20Medical%20Imaging-purple)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

## 1. Project Overview

**Brain-AI** is a deep learning project for brain tumor detection, segmentation, and visualization using brain MRI images.  
The project was developed as a research and educational prototype to explore how medical imaging AI can support tumor localization, segmentation, and explainable visualization.

The system combines:

- **YOLOv8-Segmentation** for 2D tumor detection and mask prediction
- **3D U-Net / MONAI-based workflow** for volumetric medical image segmentation experiments
- **Streamlit** for an interactive web interface
- **Plotly 3D visualization** for anatomical localization of predicted tumor regions
- **Explainable AI-style visualization** to make model predictions easier to interpret

> This project is for research and educational demonstration only. It is not a certified medical device and must not be used as a standalone diagnostic system.

---

## 2. Main Features

### 2.1 MRI Slice Upload

Users can upload a 2D MRI slice image in `.png`, `.jpg`, or `.jpeg` format through the Streamlit sidebar.

### 2.2 Brain Tumor Segmentation

The app loads a trained YOLOv8 segmentation model from `best.pt` and performs tumor region prediction on the uploaded MRI image.

The output includes:

- Predicted tumor mask
- Bounding box visualization
- Confidence score
- Tumor localization on the input image

### 2.3 3D Anatomical Localization

The detected tumor center is mapped to a normalized coordinate system and visualized in a simplified 3D brain-like structure using Plotly.

This 3D view is intended to help users understand the approximate location of the detected lesion in a more intuitive way.

### 2.4 Precision Diagnostic Interpretation Panel

The Streamlit interface displays an interpretation panel including:

- Model confidence score
- Estimated 2D lesion area in pixels
- Approximate spatial mapping
- Inference time
- Technical model parameters

### 2.5 Experimental 3D U-Net Workflow

The training notebook also includes a MONAI-based 3D U-Net pipeline using 3D NIfTI MRI volumes.  
The 3D workflow was explored to handle volumetric tumor segmentation using 3D convolution, Dice loss, and patch-based training.

---

## 3. Dataset

This project was developed using the **BraTS 2021 brain tumor MRI dataset**, distributed through Kaggle and related BraTS challenge sources.

The dataset includes multi-parametric MRI scans of adult glioma patients, including modalities such as:

- T1
- T1ce / T1Gd
- T2
- FLAIR
- Segmentation mask

The segmentation labels are commonly interpreted as:

- **ET**: Enhancing Tumor
- **TC**: Tumor Core
- **WT**: Whole Tumor

For the YOLOv8 experiment, 3D NIfTI volumes were converted into 2D image slices and YOLO-format polygon labels.

> Dataset files are not included in this repository. Users must download the dataset from the original source and follow the original dataset license and terms of use.

---

## 4. Model Summary

### 4.1 YOLOv8-Segmentation

The YOLOv8 segmentation model was trained for 2D tumor detection and segmentation.

Example training configuration:

```python
model = YOLO("yolov8s-seg.pt")
model.train(
    data="brats.yaml",
    epochs=5,
    imgsz=240,
    batch=16,
    device=0,
    project="BraTS_Short_Exp",
    name="2hour_run"
)
```

Reported experimental results from the project presentation:

| Metric | Result |
|---|---:|
| Box Precision | 0.84 |
| Mask Precision | 0.85 |
| Box Recall | 0.66 |
| Mask Recall | 0.54 |
| Box mAP50 | 0.64 |
| Mask mAP50 | 0.64 |
| Dice Score | 0.9099 |
| AUC | 0.9585 |

### 4.2 3D U-Net / MONAI

The 3D U-Net workflow uses MONAI transforms and patch-based 3D training.

Main components:

- `LoadImaged`
- `EnsureChannelFirstd`
- `NormalizeIntensityd`
- `RandSpatialCropd`
- MONAI `UNet`
- Dice Loss
- Adam optimizer

---

## 5. Repository Structure

Recommended repository structure:

```text
Brain-AI/
├── app.py                         # Streamlit inference and visualization app
├── requirements.txt               # Python dependencies
├── packages.txt                   # System packages for Streamlit Cloud deployment
├── README.md                      # Project documentation
├── LICENSE                        # MIT License for original source code
├── .gitignore                     # Excludes data, model weights, cache, and secrets
├── notebooks/
│   └── GBM_training_clean.ipynb    # Cleaned training notebook without tokens or large outputs
├── models/
│   └── best.pt                    # Trained YOLOv8 model weight file, not tracked by Git
└── assets/
    └── screenshots/               # Optional app screenshots or result figures
```

---

## 6. Installation

Clone the repository:

```bash
git clone https://github.com/jechoi2026-00/Brain-AI.git
cd Brain-AI
```

Create and activate a virtual environment:

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 7. Model Weight Setup

The Streamlit app expects a YOLOv8 weight file named:

```text
best.pt
```

Place the file in the project root directory:

```text
Brain-AI/best.pt
```

If `best.pt` is not available, the app will load but inference will not run.

Recommended approach:

- Do not commit large model weights directly if they exceed GitHub size limits.
- If needed, provide the model through GitHub Releases, Google Drive, or another external download link.
- Clearly state that the model was trained using external medical imaging data and may be subject to dataset license restrictions.

---

## 8. Running the Streamlit App

```bash
streamlit run app.py
```

Then open the local Streamlit URL displayed in the terminal.

Typical local address:

```text
http://localhost:8501
```

---

## 9. Deployment Notes

For Streamlit Community Cloud, keep these files in the root directory:

```text
app.py
requirements.txt
packages.txt
README.md
```

`packages.txt` is used to install system-level libraries required by OpenCV and image visualization packages.

Recommended `packages.txt`:

```text
libgl1
libglib2.0-0
libsm6
libxext6
libxrender1
```

---

## 10. Troubleshooting

### 10.1 Model File Not Found

Error example:

```text
Model file could not be loaded. Please check best.pt.
```

Solution:

- Make sure `best.pt` exists in the project root directory.
- Check that the file name exactly matches `best.pt`.

### 10.2 OpenCV Import Error

If OpenCV fails on Streamlit Cloud, use:

```text
opencv-python-headless
```

instead of:

```text
opencv-python
```

### 10.3 Kaggle Token Security

Do not upload `kaggle.json`, API tokens, or hard-coded Kaggle credentials to GitHub.

If a token was committed accidentally:

1. Revoke the exposed token from the Kaggle account settings.
2. Remove the token from the notebook or code.
3. Rewrite Git history if the token was pushed to a public repository.
4. Use environment variables or local-only files instead.

---

## 11. Limitations

This project has several limitations:

- The YOLOv8 model was trained for a short number of epochs.
- Some evaluation metrics were calculated on a limited sample size.
- The 3D visualization is an approximate anatomical mapping, not a clinically validated reconstruction.
- The model may produce false positives or false negatives.
- Clinical diagnosis requires expert review by qualified medical professionals.

---

## 12. Future Work

Potential extensions include:

- Multi-modal fusion using FLAIR, T1ce, T1, and T2 channels
- More robust 3D segmentation using MONAI or nnU-Net-style pipelines
- Grad-CAM or other explainable AI visualizations
- Federated learning for multi-hospital training without direct data sharing
- Additional brain disease detection such as stroke or neurodegenerative disease screening
- Web dashboard improvements with patient-level case management

---

## 13. License

This repository is licensed under the **MIT License**.

The MIT License applies only to the original source code written for this project.

External materials are not owned by the repository author and are not covered by this repository's MIT License, including but not limited to:

- BraTS / Kaggle dataset files
- MRI images and segmentation masks
- Pretrained YOLOv8 weights from Ultralytics
- Third-party libraries
- External papers, figures, screenshots, or challenge materials

Users must follow the original license and terms of use for each external resource.

---

## 14. Author

**Jeongeun Choi**  
GitHub: [@jechoi2026-00](https://github.com/jechoi2026-00)
