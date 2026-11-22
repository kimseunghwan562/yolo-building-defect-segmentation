# YOLOv8 기반 건축물 표면 결함 세그멘테이션



YOLOv8 Segmentation 모델을 활용하여 건축물 표면의 결함을 탐지하고 영역을 분할하는 프로젝트입니다.



DirectML 환경에서 모델 학습을 진행하고, 테스트 데이터셋을 이용해 객체 탐지 및 세그멘테이션 성능을 평가했습니다.



## 1. 프로젝트 소개



건축물 표면 이미지에서 발생하는 결함을 자동으로 탐지하고 분할하는 것을 목표로 합니다.



### 탐지 대상



| 클래스 | 설명 |

|---|---|

| crack | 균열 |

| efflorescence | 백화 |

| spall_detach | 박리 및 박락 |



### 사용 기술



- Python 3.12.16

- PyTorch 2.4.1

- Torch-DirectML 0.2.5.dev240914

- Ultralytics 8.4.212

- YOLOv8 Segmentation



## 2. 프로젝트 구조



```text

yolo-building-defect-segmentation/

├── artifacts/

│   ├── best.pt

│   └── training_log.csv

├── data/

│   └── data.yaml

├── evaluate_yolov8_seg.py

├── predict_test_images.py

├── train_yolov8_seg_dml_resume.py

├── yolov8s-seg.pt

├── .gitignore

└── README.md

```



- `artifacts/best.pt`: 최종 학습 모델

- `artifacts/training_log.csv`: 학습 과정 기록

- `data/data.yaml`: 데이터셋 및 클래스 설정

- `evaluate_yolov8_seg.py`: 테스트 데이터셋 평가

- `predict_test_images.py`: 테스트 이미지 예측

├── train_yolov8_seg_dml_resume.py

- `yolov8s-seg.pt`: 사전 학습 가중치



실제 이미지와 라벨 데이터, 학습 중간 결과 및 가상환경은 저장소에 포함하지 않았습니다.



## 3. 학습 환경 및 과정



- Python 3.12.16

- 입력 이미지 크기: 320 × 320

- Batch Size: 1

- Optimizer: AdamW

- Learning Rate: 0.0001

- Weight Decay: 0.0005

- Scheduler: CosineAnnealingLR

- 총 학습 Epoch: 50



DirectML 환경에서 학습을 진행했으며, 학습 중 일부 배치에서 오류가 발생하여 해당 배치를 건너뛰도록 처리했습니다.



최종 학습 로그에는 Epoch 3부터 Epoch 50까지 총 48개 기록이 포함되어 있습니다.



## 4. 테스트 결과



테스트 데이터셋 40장을 대상으로 평가했습니다.



### 정량적 평가 결과



| 평가 지표 | 결과 |

|---|---:|

| Box Precision | 63.74% |

| Box Recall | 4.76% |

| Box mAP50 | 2.83% |

| Box mAP50-95 | 0.83% |

| Mask mAP50 | 3.14% |

| Mask mAP50-95 | 1.72% |



### 탐지 결과 요약



- 전체 테스트 이미지: 40장

- 탐지가 발생한 이미지: 4장

- 탐지된 객체: 15개

- 탐지가 발생하지 않은 이미지: 36장



현재 모델은 일부 이미지에서 결함 영역을 탐지했지만, 전체 테스트 데이터에 대한 재현율과 mAP가 낮아 안정적인 결함 탐지 성능을 확보했다고 보기는 어렵습니다.



### 대표 탐지 사례



아래는 실제 테스트 이미지 중 모델이 객체를 탐지한 4개 사례입니다.

### 사례 1

<p align="center">
  <img src="https://github.com/user-attachments/assets/6be9d521-eb6c-4484-85da-1fe60ccd7a47" width="49%" alt="사례 1 원본 이미지" />
  <img src="https://github.com/user-attachments/assets/cd232c92-7747-4393-bf4a-c840f233a5d5" width="49%" alt="사례 1 예측 결과" />
</p>

### 사례 2

<p align="center">
  <img src="https://github.com/user-attachments/assets/1f144360-befb-425d-b748-ebb77ec2cbbb" width="49%" alt="사례 2 원본 이미지" />
  <img src="https://github.com/user-attachments/assets/75228e40-c68a-45d9-a1d4-8551bbc36f5b" width="49%" alt="사례 2 예측 결과" />
</p>

### 사례 3

<p align="center">
  <img src="https://github.com/user-attachments/assets/bf2519ac-337c-40a1-93a8-29f7b4919b96" width="49%" alt="사례 3 원본 이미지" />
  <img src="https://github.com/user-attachments/assets/f5032c35-2ec4-4da2-bf19-f669d1d5b23c" width="49%" alt="사례 3 예측 결과" />
</p>

### 사례 4

<p align="center">
  <img src="https://github.com/user-attachments/assets/643ed828-3809-4556-9479-3b35a6078ec3" width="49%" alt="사례 4 원본 이미지" />
  <img src="https://github.com/user-attachments/assets/ea5d301f-5955-4a8f-9941-51329d9ff01b" width="49%" alt="사례 4 예측 결과" />
</p>





















위 사례는 실제 탐지가 발생한 이미지의 정성적 예시이며, 전체 테스트 성능을 대표한다고 볼 수는 없습니다.



## 5. 실행 방법



### 데이터셋 구성



실행을 위해 별도의 데이터셋이 필요합니다.



```text

data/

├── data.yaml

├── images/

│   ├── train/

│   ├── val/

│   └── test/

└── labels/

    ├── train/

    ├── val/

    └── test/

```



데이터셋은 저장소에 포함되어 있지 않습니다.



### 테스트 데이터 평가



```powershell

python evaluate_yolov8_seg.py

```



### 이미지 예측



```powershell

python predict_test_images.py

```



평가 및 예측 코드는 `artifacts/best.pt`를 사용합니다.



현재 최종 체크포인트는 일반적인 Ultralytics 모델 파일과 저장 구조가 다르므로, 코드에서 모델 아키텍처를 생성한 뒤 `model_state_dict`를 불러오는 방식을 사용합니다.



> 위 실행 명령은 데이터셋과 필요한 Python 패키지가 준비된 환경을 전제로 합니다.



## 6. 한계 및 개선 방향



현재 프로젝트의 주요 한계는 다음과 같습니다.



- 테스트 데이터에서 탐지에 성공한 이미지가 40장 중 4장으로 제한적입니다.

- Box 및 Mask mAP가 낮아 결함 탐지 성능 개선이 필요합니다.

- 클래스별 탐지 성능에 차이가 있습니다.

- 일부 학습 배치가 오류로 인해 건너뛰어졌습니다.

- 데이터셋의 다양성과 학습 안정성을 추가로 검토할 필요가 있습니다.



향후 개선 방향은 다음과 같습니다.



- 결함 유형별 데이터 분포 및 라벨 품질 점검

- 클래스별 탐지 성능 개선

- 학습 과정의 안정성 향상

- 테스트 데이터 확대를 통한 일반화 성능 검증



## 7. 프로젝트 요약



YOLOv8 Segmentation 모델을 이용해 건축물 표면 결함 3종을 탐지하고 분할하는 프로젝트입니다.



DirectML 환경에서 모델 학습 과정을 구현하고, 테스트 데이터셋을 이용한 정량적 평가와 실제 예측 결과 분석을 진행했습니다.



현재 성능에는 한계가 있지만, 학습 환경 구성부터 모델 학습, 체크포인트 관리, 테스트 평가 및 결과 시각화까지의 과정을 경험하고 구현한 프로젝트입니다.
