import torch
from ultralytics import YOLO
from ultralytics.nn.tasks import SegmentationModel

checkpoint_path = "artifacts/best.pt"

model = YOLO("yolov8s-seg.yaml")
model.model = SegmentationModel(
    cfg="yolov8s-seg.yaml",
    ch=3,
    nc=3,
    verbose=False
)

checkpoint = torch.load(
    checkpoint_path,
    map_location="cpu",
    weights_only=False
)
model.model.load_state_dict(checkpoint["model_state_dict"], strict=True)
model.model.eval()

results = model.predict(
    source="data/images/test",
    imgsz=320,
    conf=0.25,
    device="cpu",
    save=True,
    save_txt=True,
    project="runs_predict",
    name="test_all",
    exist_ok=True
)

detected = sum(len(result.boxes) for result in results)
print("\n전체 이미지:", len(results))
print("탐지된 이미지:", sum(len(result.boxes) > 0 for result in results))
print("탐지 객체 수:", detected)
print("결과 저장 위치: runs_predict/test_all")
