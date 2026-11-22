import torch
from pathlib import Path
from ultralytics import YOLO
from ultralytics.nn.tasks import SegmentationModel

checkpoint_path = Path(
    "artifacts/best.pt"
)

print("체크포인트 불러오는 중...")
checkpoint = torch.load(
    checkpoint_path,
    map_location="cpu",
    weights_only=False
)

print("모델 생성 중...")
model = YOLO("yolov8s-seg.yaml")

model.model = SegmentationModel(
    cfg="yolov8s-seg.yaml",
    ch=3,
    nc=3,
    verbose=False
)

model.model.load_state_dict(
    checkpoint["model_state_dict"],
    strict=True
)

model.model.eval()

imgsz = checkpoint.get("imgsz", 320)

print("모델:", checkpoint["model_name"])
print("Epoch:", checkpoint["epoch"])
print("평가 이미지 크기:", imgsz)
print("테스트 데이터 평가 시작")

results = model.val(
    data="data/data.yaml",
    split="test",
    imgsz=imgsz,
    batch=1,
    device="cpu",
    workers=0,
    plots=True,
    project="runs_eval",
    name="final_test",
    exist_ok=True
)

print("\n===== 테스트 평가 결과 =====")
print("Box Precision:", results.box.mp)
print("Box Recall:", results.box.mr)
print("Box mAP50:", results.box.map50)
print("Box mAP50-95:", results.box.map)
print("Mask mAP50:", results.seg.map50)
print("Mask mAP50-95:", results.seg.map)
