# Brain-AI

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![Jupyter Notebook](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Web_App-FF4B4B?logo=streamlit&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep_Learning-EE4C2C?logo=pytorch&logoColor=white)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Object_Detection-00FFFF)
![OpenCV](https://img.shields.io/badge/OpenCV-Image_Processing-5C3EE8?logo=opencv&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-Visualization-3F4F75?logo=plotly&logoColor=white)

## 프로젝트 소개

**Brain-AI**는 뇌 MRI 이미지에서 뇌종양 또는 병변 의심 영역을 탐지하고, 탐지 결과를 시각적으로 해석할 수 있도록 구성한 딥러닝 기반 의료영상 분석 프로토타입입니다.

사용자가 MRI 슬라이스 이미지를 업로드하면, Streamlit 웹 애플리케이션이 YOLO 기반 모델을 이용해 병변 의심 영역을 탐지하고, 탐지 신뢰도, 2D segmentation 결과, 3D anatomical localization, 정밀 진단 리포트를 함께 제공합니다.

> ⚠️ 본 프로젝트는 연구 및 포트폴리오 목적의 의료 AI 프로토타입입니다. 실제 임상 진단, 치료 결정, 수술 계획, 의료 판단을 대체할 수 없습니다. 최종 판단은 반드시 전문 의료진의 판독을 따라야 합니다.

---

## 주요 기능

- MRI 이미지 업로드 기반 뇌종양 의심 영역 탐지
- YOLO 기반 딥러닝 모델 추론
- 병변 의심 영역 bounding box / segmentation 시각화
- 탐지 신뢰도(confidence score) 출력
- Pixel intensity, morphology 기반 AI 탐지 근거 설명
- Plotly 기반 3D anatomical localization 시각화
- 병변 위치를 좌우/전후 방향으로 정규화하여 표시
- 병변 단면적 기반 estimated size 계산
- Streamlit 기반 웹 프로토타입 제공

---

## 프로젝트 구조

```text
Brain-AI/
├── .gitignore
├── GBM.ipynb
├── README.md
├── app.py
├── packages.txt
└── requirements.txt
```

> 실행을 위해서는 학습된 YOLO 모델 파일인 `best.pt`가 프로젝트 루트에 필요합니다. 현재 `app.py`는 `YOLO('best.pt')`를 불러오도록 작성되어 있습니다.

---

## 파일 설명

| 파일명 | 설명 |
|---|---|
| `.gitignore` | 가상환경 폴더 등 불필요한 파일이 GitHub에 올라가지 않도록 제외하는 설정 파일 |
| `GBM.ipynb` | 뇌종양/GBM 관련 모델 학습 또는 실험 과정을 정리한 Jupyter Notebook 파일 |
| `README.md` | 프로젝트 소개, 설치 방법, 실행 방법, 파일 구조를 설명하는 문서 |
| `app.py` | Streamlit 웹 애플리케이션 메인 실행 파일 |
| `packages.txt` | Streamlit Cloud 등 배포 환경에서 필요한 Linux 시스템 패키지 목록 |
| `requirements.txt` | Python 실행 환경에 필요한 패키지 목록 |
| `best.pt` | YOLO 모델 가중치 파일. 앱 실행 시 필요하지만, 저장소에 없을 경우 별도로 추가해야 함 |

---

## 기술 스택

| 구분 | 사용 기술 | 역할 |
|---|---|---|
| Web App | Streamlit | MRI 이미지 업로드 및 결과 화면 구성 |
| Deep Learning | PyTorch | 모델 추론 기반 프레임워크 |
| Object Detection / Segmentation | Ultralytics YOLO | 병변 의심 영역 탐지 |
| Image Processing | OpenCV, Pillow | 이미지 로딩, 변환, 전처리 |
| Numerical Computing | NumPy, Pandas | 좌표 계산 및 결과 처리 |
| Visualization | Plotly | 3D localization 및 진단 리포트 시각화 |
| Experiment | Jupyter Notebook | 모델 실험 및 분석 과정 정리 |

---

## 분석 흐름

```text
MRI 이미지 업로드
        ↓
PIL / OpenCV 기반 이미지 로드 및 RGB 변환
        ↓
YOLO 모델 추론
        ↓
병변 의심 영역 탐지
        ↓
2D segmentation 결과 표시
        ↓
신뢰도, 병변 크기, 위치 좌표 계산
        ↓
3D anatomical localization 시각화
        ↓
정밀 진단 리포트 출력
```

---

## 실행 전 준비 사항

### 1. 모델 파일 확인

`app.py`는 아래 코드처럼 `best.pt` 파일을 불러옵니다.

```python
model = YOLO('best.pt')
```

따라서 실행 전 프로젝트 루트에 `best.pt` 파일이 있어야 합니다.

```text
Brain-AI/
├── app.py
├── best.pt
├── requirements.txt
└── ...
```

만약 모델 파일명이 다르다면 `app.py`에서 아래 부분을 실제 파일명에 맞게 수정해야 합니다.

```python
model = YOLO('your_model_name.pt')
```

---

## 설치 방법

### 1. 저장소 클론

```bash
git clone https://github.com/jechoi2026-00/Brain-AI.git
cd Brain-AI
```

### 2. 가상환경 생성

Windows 기준:

```bash
python -m venv .venv
.venv\Scripts\activate
```

macOS / Linux 기준:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. 패키지 설치

```bash
pip install -r requirements.txt
```

---

## 실행 방법

아래 명령어로 Streamlit 앱을 실행합니다.

```bash
streamlit run app.py
```

실행 후 브라우저에서 앱 화면이 열리면 사이드바에서 MRI 이미지를 업로드할 수 있습니다.

---

## 입력 데이터 형식

현재 웹 앱은 다음 형식의 이미지 업로드를 지원합니다.

| 항목 | 내용 |
|---|---|
| 입력 형식 | `.png`, `.jpg`, `.jpeg` |
| 입력 데이터 | MRI 슬라이스 이미지 |
| 예시 모달리티 | T1CE, T2, FLAIR 등 |
| 출력 | 병변 탐지 결과, 신뢰도, 위치 시각화, 진단 보조 리포트 |

> 현재 앱은 이미지 파일 기반 프로토타입입니다. DICOM 또는 NIfTI(`.nii.gz`) 원본 의료영상 전체 볼륨을 직접 처리하는 구조는 아닙니다.

---

## 결과 화면 구성

### 1. 2D Segmentation & XAI

업로드한 MRI 이미지 위에 모델이 탐지한 병변 의심 영역을 표시합니다.

출력 항목:

- AI-segmented MRI slice
- 병변 감지 여부
- 탐지 신뢰도
- Pixel intensity 기반 설명
- Morphology 기반 설명

### 2. Anatomical 3D Localization

탐지된 병변의 중심 좌표를 이미지 크기 기준으로 정규화한 뒤, Plotly 기반 3D 뇌 윤곽 시뮬레이션 위에 표시합니다.

출력 항목:

- 좌우 방향 위치
- 전후 방향 위치
- 종양 중심 좌표
- 3D 시각화 그래프

### 3. Precision Diagnostic Interpretation

탐지 결과를 리포트 형태로 요약합니다.

출력 항목:

- Confidence Score
- Estimated Size
- Spatial Mapping
- Inference Time
- Preprocessing 정보
- Analysis Status

---

## 핵심 코드 구조

### 모델 로드

```python
@st.cache_resource
def load_yolo_model():
    model = YOLO('best.pt')
    return model
```

### 이미지 업로드

```python
uploaded_file = st.sidebar.file_uploader(
    "MRI 이미지 업로드 (T1CE/T2/Flair)",
    type=["png", "jpg", "jpeg"]
)
```

### 모델 추론

```python
results = model(img_array)
```

### 병변 탐지 결과 표시

```python
if len(results[0].boxes) > 0:
    res_plotted = results[0].plot()
    st.image(res_plotted, use_container_width=True)
```

---

## 배포 관련 참고

Streamlit Cloud 배포 시에는 다음 파일들이 필요합니다.

```text
requirements.txt
packages.txt
app.py
best.pt
```

`packages.txt`에는 OpenCV 실행에 필요한 Linux 시스템 라이브러리가 포함되어 있습니다.

```text
libgl1
libglib2.0-0
libsm6
libxext6
libxrender1
```

모델 파일인 `best.pt`의 용량이 큰 경우 GitHub 일반 업로드 대신 다음 방식을 고려할 수 있습니다.

- Git LFS 사용
- Hugging Face Hub에 모델 업로드 후 앱에서 다운로드
- Streamlit Secrets 또는 외부 스토리지 활용
- Release asset으로 모델 파일 관리

---

## 주의 사항

- 본 프로젝트는 의료 AI 연구 및 포트폴리오용 프로토타입입니다.
- 실제 환자 데이터나 개인정보가 포함된 의료영상은 Public 저장소에 업로드하면 안 됩니다.
- `best.pt` 모델 파일이 없으면 앱이 정상적으로 실행되지 않습니다.
- 모델 성능은 학습 데이터, 전처리 방식, 입력 이미지 품질에 따라 달라질 수 있습니다.
- 현재 3D localization은 MRI 전체 볼륨 기반 정밀 3D 재구성이 아니라, 2D 이미지 탐지 좌표를 정규화하여 시각화하는 프로토타입 방식입니다.

---

## Disclaimer

This project is for research and educational purposes only. It is not intended for clinical diagnosis, treatment planning, or medical decision-making. Any medical interpretation must be confirmed by qualified healthcare professionals.
