import csv
import copy
import random
import shutil
from pathlib import Path
from datetime import datetime

import torch
import torch_directml
import types

from torch.utils.data import DataLoader
from ultralytics import YOLO
from ultralytics.nn.tasks import SegmentationModel
from ultralytics.cfg import get_cfg
from ultralytics.utils import DEFAULT_CFG
from ultralytics.data import build_yolo_dataset
from ultralytics.utils.ops import xywh2xyxy


# ============================================================
# 1. 기본 설정
# ============================================================

EPOCHS = 50
IMAGE_SIZE = 320
BATCH_SIZE = 1
LEARNING_RATE = 0.0001
WEIGHT_DECAY = 0.0005
SEED = 42

random.seed(SEED)
torch.manual_seed(SEED)

device = torch_directml.device()

print("=" * 60)
print("YOLOv8 Segmentation DirectML 실제 학습")
print("장치:", device)
print("Epoch:", EPOCHS)
print("Image size:", IMAGE_SIZE)
print("Batch size:", BATCH_SIZE)
print("=" * 60)


# ============================================================
# 2. 모델 생성 및 사전학습 가중치 적용
# ============================================================

print("\n[1/6] 모델 생성")

model = YOLO("yolov8s-seg.yaml")

model.model = SegmentationModel(
    cfg="yolov8s-seg.yaml",
    ch=3,
    nc=3,
    verbose=False
)

model.load("yolov8s-seg.pt")

net = model.model

net.args = get_cfg(DEFAULT_CFG, {
    "task": "segment",
    "imgsz": IMAGE_SIZE,
    "overlap_mask": True
})

net = net.to(device)
net.train()

print("클래스 수:", net.model[-1].nc)
print("학습 파라미터:", sum(
    p.numel() for p in net.parameters() if p.requires_grad
))


# ============================================================
# 3. 데이터셋 생성
# ============================================================

print("\n[2/6] 데이터셋 생성")

data_info = {
    "names": {
        0: "crack",
        1: "efflorescence",
        2: "spall_detach"
    },
    "nc": 3,
    "channels": 3
}

train_cfg = get_cfg(DEFAULT_CFG, {
    "task": "segment",
    "imgsz": IMAGE_SIZE,
    "batch": BATCH_SIZE,
    "rect": False,
    "mode": "train",
    "augment": False,
    "data": "data/data.yaml"
})

val_cfg = get_cfg(DEFAULT_CFG, {
    "task": "segment",
    "imgsz": IMAGE_SIZE,
    "batch": BATCH_SIZE,
    "rect": False,
    "mode": "val",
    "augment": False,
    "data": "data/data.yaml"
})

train_dataset = build_yolo_dataset(
    cfg=train_cfg,
    img_path="data/images/train",
    batch=BATCH_SIZE,
    data=data_info,
    mode="train"
)

val_dataset = build_yolo_dataset(
    cfg=val_cfg,
    img_path="data/images/val",
    batch=BATCH_SIZE,
    data=data_info,
    mode="val"
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=False,
    collate_fn=train_dataset.collate_fn
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=False,
    collate_fn=val_dataset.collate_fn
)

print("Train 이미지:", len(train_dataset))
print("Validation 이미지:", len(val_dataset))
print("Train 배치:", len(train_loader))
print("Validation 배치:", len(val_loader))


# ============================================================
# 4. DirectML 호환 Loss 수정
# ============================================================

print("\n[3/6] DirectML 호환 Loss 구성")

criterion = net.init_criterion()


def preprocess_single_image(self, targets, batch_size, scale_tensor):

    nl, ne = targets.shape

    if batch_size != 1:
        raise RuntimeError("batch_size=1 전용 함수입니다.")

    out = torch.zeros(
        (1, nl, ne - 1),
        device=self.device,
        dtype=targets.dtype
    )

    if nl == 0:
        return out

    out[0] = targets[:, 1:]

    out[..., 1:5] = xywh2xyxy(
        out[..., 1:5] * scale_tensor
    )

    return out


def select_topk_candidates_dml(self, metrics, topk_mask=None):

    topk_metrics, topk_idxs = torch.topk(
        metrics,
        self.topk,
        dim=-1,
        largest=True
    )

    if topk_mask is None:
        topk_mask = (
            topk_metrics.max(-1, keepdim=True)[0] > self.eps
        ).expand_as(topk_idxs)

    topk_idxs = topk_idxs.masked_fill(~topk_mask, 0)

    anchor_indices = torch.arange(
        metrics.shape[-1],
        device=metrics.device
    ).view(1, 1, 1, -1)

    selected = topk_idxs.unsqueeze(-1) == anchor_indices

    count_tensor = selected.sum(dim=-2).to(torch.int8)

    count_tensor.masked_fill_(count_tensor > 1, 0)

    return count_tensor


def assigner_forward_dml(
    self,
    pd_scores,
    pd_bboxes,
    anc_points,
    gt_labels,
    gt_bboxes,
    mask_gt,
    **extra
):

    mask_pos, align_metric, overlaps = self.get_pos_mask(
        pd_scores,
        pd_bboxes,
        gt_labels,
        gt_bboxes,
        anc_points,
        mask_gt,
        **extra
    )

    target_gt_idx, fg_mask, mask_pos = self.select_highest_overlaps(
        mask_pos,
        overlaps,
        self.n_max_boxes,
        align_metric
    )

    target_labels, target_bboxes, target_scores = self.get_targets(
        gt_labels,
        gt_bboxes,
        target_gt_idx,
        fg_mask
    )

    target_scores = target_scores.to(dtype=torch.float32)

    mask_float = mask_pos.to(dtype=align_metric.dtype)

    align_metric = align_metric * mask_float

    pos_align_metrics = align_metric.amax(
        dim=-1,
        keepdim=True
    )

    overlaps = overlaps * mask_float

    pos_overlaps = overlaps.amax(
        dim=-1,
        keepdim=True
    )

    align_metric = (
        align_metric * pos_overlaps
    ) / (
        pos_align_metrics + self.eps
    )

    norm_align_metric = align_metric.amax(
        -2
    ).unsqueeze(-1)

    target_scores = target_scores * norm_align_metric

    return (
        target_labels,
        target_bboxes,
        target_scores,
        fg_mask.bool(),
        target_gt_idx
    )


def single_mask_loss_dml(gt_mask, pred, proto, xyxy, area):

    pred_mask = torch.einsum(
        "in,nhw->ihw",
        pred,
        proto
    )

    gt_mask = gt_mask.to(dtype=pred_mask.dtype)

    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        pred_mask,
        gt_mask,
        reduction="none"
    )

    _, h, w = loss.shape

    x1, y1, x2, y2 = xyxy.unbind(dim=1)

    r = torch.arange(
        w,
        device=loss.device,
        dtype=xyxy.dtype
    ).view(1, 1, w)

    c = torch.arange(
        h,
        device=loss.device,
        dtype=xyxy.dtype
    ).view(1, h, 1)

    crop = (
        (r >= x1.view(-1, 1, 1))
        & (r < x2.view(-1, 1, 1))
        & (c >= y1.view(-1, 1, 1))
        & (c < y2.view(-1, 1, 1))
    ).to(dtype=loss.dtype)

    cropped_loss = loss * crop

    return (
        cropped_loss.mean(dim=(1, 2)) / area
    ).sum()




def dfl_loss_forward_dml(pred_dist, target, reg_max):
    if target.numel() == 0:
        return pred_dist[:, :1] * 0.0

    # 기존 DFLoss와 동일한 계산. gather() 대신 비교 마스크를 사용한다.
    target = target.clamp(0, reg_max - 1 - 0.01)

    tl = target.long()
    tr = tl + 1

    wl = tr.to(dtype=target.dtype) - target
    wr = 1.0 - wl

    logp = torch.nn.functional.log_softmax(pred_dist, dim=1)

    bins = torch.arange(
        reg_max,
        device=pred_dist.device,
    ).view(1, -1)

    left_mask = (
        bins == tl.reshape(-1, 1)
    ).to(dtype=logp.dtype)

    right_mask = (
        bins == tr.reshape(-1, 1)
    ).to(dtype=logp.dtype)

    left_logp = (
        logp * left_mask
    ).sum(dim=1).reshape(tl.shape)

    right_logp = (
        logp * right_mask
    ).sum(dim=1).reshape(tr.shape)

    return -(
        left_logp * wl + right_logp * wr
    ).mean(-1, keepdim=True)


def bbox_loss_forward_dml(
    self,
    pred_dist,
    pred_bboxes,
    anchor_points,
    target_bboxes,
    target_scores,
    target_scores_sum,
    fg_mask,
    imgsz,
    stride,
):
    from ultralytics.utils.tal import bbox2dist

    # 기존 Ultralytics 방식으로 양성 위치를 선택한다.
    fg_idx = fg_mask.nonzero(as_tuple=True)

    pred_selected = pred_bboxes[fg_idx]
    target_selected = target_bboxes[fg_idx]
    weight = target_scores[fg_idx].sum(-1, keepdim=True)

    # bbox_iou()의 chunk()를 사용하지 않고 좌표별로 계산한다.
    px1 = pred_selected[:, 0:1]
    py1 = pred_selected[:, 1:2]
    px2 = pred_selected[:, 2:3]
    py2 = pred_selected[:, 3:4]

    tx1 = target_selected[:, 0:1]
    ty1 = target_selected[:, 1:2]
    tx2 = target_selected[:, 2:3]
    ty2 = target_selected[:, 3:4]

    eps = 1e-7

    pw = (px2 - px1).clamp(min=0)
    ph = (py2 - py1).clamp(min=0)
    tw = (tx2 - tx1).clamp(min=0)
    th = (ty2 - ty1).clamp(min=0)

    inter_w = (
        torch.minimum(px2, tx2) - torch.maximum(px1, tx1)
    ).clamp(min=0)

    inter_h = (
        torch.minimum(py2, ty2) - torch.maximum(py1, ty1)
    ).clamp(min=0)

    intersection = inter_w * inter_h
    union = pw * ph + tw * th - intersection + eps
    iou = intersection / union

    # CIoU의 중심점 거리 항
    pcx = (px1 + px2) / 2
    pcy = (py1 + py2) / 2
    tcx = (tx1 + tx2) / 2
    tcy = (ty1 + ty2) / 2

    center_distance = (pcx - tcx).pow(2) + (pcy - tcy).pow(2)

    enclosing_x1 = torch.minimum(px1, tx1)
    enclosing_y1 = torch.minimum(py1, ty1)
    enclosing_x2 = torch.maximum(px2, tx2)
    enclosing_y2 = torch.maximum(py2, ty2)

    enclosing_w = (enclosing_x2 - enclosing_x1).clamp(min=0)
    enclosing_h = (enclosing_y2 - enclosing_y1).clamp(min=0)
    enclosing_diagonal = enclosing_w.pow(2) + enclosing_h.pow(2) + eps

    # CIoU의 종횡비 항
    v = (4.0 / (torch.pi ** 2)) * (
        torch.atan(tw / (th + eps)) -
        torch.atan(pw / (ph + eps))
    ).pow(2)

    with torch.no_grad():
        alpha = v / (v - iou + 1.0 + eps)

    ciou = iou - (
        center_distance / enclosing_diagonal + alpha * v
    )

    loss_iou = ((1.0 - ciou) * weight).sum() / target_scores_sum

    # Distribution Focal Loss
    if self.dfl_loss:
        target_ltrb = bbox2dist(
            anchor_points,
            target_bboxes,
            self.dfl_loss.reg_max - 1,
        )

        selected_pred_dist = pred_dist[fg_idx].reshape(
            -1, self.dfl_loss.reg_max
        )
        selected_target_ltrb = target_ltrb[fg_idx]

        loss_dfl = (
            dfl_loss_forward_dml(selected_pred_dist, selected_target_ltrb, self.dfl_loss.reg_max)
            * weight
        )
        loss_dfl = loss_dfl.sum() / target_scores_sum

    else:
        target_ltrb = bbox2dist(anchor_points, target_bboxes)

        target_ltrb = target_ltrb * stride
        target_ltrb[..., 0::2] /= imgsz[1]
        target_ltrb[..., 1::2] /= imgsz[0]

        pred_dist_scaled = pred_dist * stride
        pred_dist_scaled[..., 0::2] /= imgsz[1]
        pred_dist_scaled[..., 1::2] /= imgsz[0]

        loss_dfl = (
            torch.nn.functional.l1_loss(
                pred_dist_scaled[fg_idx],
                target_ltrb[fg_idx],
                reduction="none",
            ).mean(-1, keepdim=True)
            * weight
        )
        loss_dfl = loss_dfl.sum() / target_scores_sum

    return loss_iou, loss_dfl


criterion.preprocess = types.MethodType(
    preprocess_single_image,
    criterion
)

criterion.assigner.select_topk_candidates = types.MethodType(
    select_topk_candidates_dml,
    criterion.assigner
)

criterion.assigner._forward = types.MethodType(
    assigner_forward_dml,
    criterion.assigner
)

criterion.single_mask_loss = single_mask_loss_dml

net.criterion = criterion
criterion.bbox_loss.forward = types.MethodType(
    bbox_loss_forward_dml,
    criterion.bbox_loss,
)


print("DirectML Loss 호환 수정 완료")


# ============================================================
# 5. Optimizer 및 저장 경로
# ============================================================

print("\n[4/6] Optimizer 설정")

optimizer = torch.optim.AdamW(
    net.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=EPOCHS,
    eta_min=0.000001
)

# ?? Epoch 2 ????? ??
resume_path = Path('runs_directml/yolov8s_seg_20261121_180617/weights/last.pt')
best_path = Path('runs_directml/yolov8s_seg_20261121_180617/weights/best.pt')

resume_checkpoint = torch.load(
    resume_path, map_location='cpu', weights_only=False
)
best_checkpoint = torch.load(
    best_path, map_location='cpu', weights_only=False
)

net.load_state_dict(resume_checkpoint['model_state_dict'], strict=True)
optimizer.load_state_dict(resume_checkpoint['optimizer_state_dict'])
scheduler.load_state_dict(resume_checkpoint['scheduler_state_dict'])

start_epoch = resume_checkpoint['epoch']
best_val_loss = best_checkpoint['val_loss']

print('\n=== ????? ?? ?? ===')
print('?? Epoch:', start_epoch)
print('Best Validation Loss:', best_val_loss)
print('?? ?? Epoch:', start_epoch + 1)

run_dir = (
    Path("runs_directml")
    / datetime.now().strftime("yolov8s_seg_%Y%m%d_%H%M%S")
)

run_dir.mkdir(parents=True, exist_ok=True)

weights_dir = run_dir / "weights"
weights_dir.mkdir(exist_ok=True)

# ?? Best ?????? ? ?? ??? ??
shutil.copy2(best_path, weights_dir / 'best.pt')

log_path = run_dir / "training_log.csv"

print("저장 위치:", run_dir.resolve())


def to_cpu(obj):

    if torch.is_tensor(obj):
        return obj.detach().cpu()

    if isinstance(obj, dict):
        return {k: to_cpu(v) for k, v in obj.items()}

    if isinstance(obj, list):
        return [to_cpu(v) for v in obj]

    if isinstance(obj, tuple):
        return tuple(to_cpu(v) for v in obj)

    return obj


def save_checkpoint(epoch, train_loss, val_loss):

    model_state = {
        k: v.detach().cpu()
        for k, v in net.state_dict().items()
    }

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model_state,
        "optimizer_state_dict": to_cpu(optimizer.state_dict()),
        "scheduler_state_dict": scheduler.state_dict(),
        "train_loss": train_loss,
        "val_loss": val_loss,
        "class_names": data_info["names"],
        "imgsz": IMAGE_SIZE,
        "model_name": "yolov8s-seg",
    }

    torch.save(
        checkpoint,
        weights_dir / "last.pt"
    )


def save_best(epoch, train_loss, val_loss):

    model_state = {
        k: v.detach().cpu()
        for k, v in net.state_dict().items()
    }

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model_state,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "class_names": data_info["names"],
            "imgsz": IMAGE_SIZE,
            "model_name": "yolov8s-seg",
        },
        weights_dir / "best.pt"
    )


# ============================================================
# 6. 학습 및 검증 함수
# ============================================================

def move_batch_to_device(batch):

    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            batch[key] = value.to(device)

    batch["img"] = batch["img"].float() / 255.0

    return batch


def validate():

    net.train()

    # 검증 중 BatchNorm 통계가 변경되지 않도록 설정
    for module in net.modules():
        if isinstance(
            module,
            torch.nn.modules.batchnorm._BatchNorm
        ):
            module.eval()

    total_loss = 0.0
    valid_batches = 0
    skipped_batches = 0

    with torch.no_grad():

        for batch in val_loader:

            if batch["cls"].shape[0] == 0:
                skipped_batches += 1
                continue

            batch = move_batch_to_device(batch)

            loss_vector, _ = net.loss(batch)

            if not torch.isfinite(loss_vector).all().item():
                raise RuntimeError(
                    "Validation Loss에 NaN 또는 Inf가 발생했습니다."
                )

            total_loss += loss_vector.sum().item()
            valid_batches += 1

    net.train()

    if valid_batches == 0:
        raise RuntimeError(
            "검증 데이터에서 객체가 포함된 배치를 찾지 못했습니다."
        )

    return (
        total_loss / valid_batches,
        valid_batches,
        skipped_batches
    )


# ============================================================
# 7. 실제 학습 시작
# ============================================================

print("\n[5/6] 실제 학습 시작")

# ??? Best Validation Loss ??

with open(
    log_path,
    "w",
    newline="",
    encoding="utf-8-sig"
) as log_file:

    writer = csv.writer(log_file)

    writer.writerow([
        "epoch",
        "train_loss",
        "val_loss",
        "learning_rate",
        "successful_batches",
        "skipped_batches"
    ])

    for epoch in range(start_epoch + 1, EPOCHS + 1):

        net.train()

        epoch_loss = 0.0
        successful_steps = 0
        skipped_batches = 0

        print("\n" + "=" * 60)
        print(f"Epoch {epoch}/{EPOCHS} 시작")
        print("=" * 60)

        for step, batch in enumerate(train_loader, start=1):

            # 객체가 없는 이미지는 현재 DirectML 호환 경로에서 제외
            if batch["cls"].shape[0] == 0:
                skipped_batches += 1
                continue

            batch = move_batch_to_device(batch)

            optimizer.zero_grad(set_to_none=True)

            loss_vector, loss_items = net.loss(batch)

            if not torch.isfinite(loss_vector).all().item():
                raise RuntimeError(
                    f"Epoch {epoch}, Step {step}: "
                    "Loss에 NaN 또는 Inf가 발생했습니다."
                )

            loss_total = loss_vector.sum()

            loss_total.backward()

            grad_count = sum(
                1
                for p in net.parameters()
                if p.requires_grad and p.grad is not None
            )

            if grad_count == 0:
                raise RuntimeError(
                    "Gradient가 생성되지 않았습니다."
                )

            optimizer.step()

            epoch_loss += loss_total.detach().item()
            successful_steps += 1

            if step % 100 == 0 or step == len(train_loader):

                print(
                    f"Epoch {epoch}/{EPOCHS} | "
                    f"Batch {step}/{len(train_loader)} | "
                    f"Loss {loss_total.item():.5f} | "
                    f"Gradient {grad_count}"
                )

        if successful_steps == 0:
            raise RuntimeError(
                "이번 Epoch에서 학습 가능한 배치가 없습니다."
            )

        train_avg = epoch_loss / successful_steps

        # 검증
        val_avg, val_steps, val_skipped = validate()

        current_lr = optimizer.param_groups[0]["lr"]

        print("\nEpoch 결과")
        print(f"Train Loss: {train_avg:.6f}")
        print(f"Validation Loss: {val_avg:.6f}")
        print(f"학습 배치: {successful_steps}")
        print(f"학습 제외 배치: {skipped_batches}")
        print(f"검증 배치: {val_steps}")
        print(f"검증 제외 배치: {val_skipped}")
        print(f"Learning Rate: {current_lr:.8f}")

        scheduler.step()

        # 매 Epoch 저장
        save_checkpoint(
            epoch,
            train_avg,
            val_avg
        )

        # 검증 Loss 기준 Best 저장
        if val_avg < best_val_loss:

            best_val_loss = val_avg

            save_best(
                epoch,
                train_avg,
                val_avg
            )

            print("Best 모델 갱신 완료")

        writer.writerow([
            epoch,
            train_avg,
            val_avg,
            current_lr,
            successful_steps,
            skipped_batches
        ])

        log_file.flush()

        print(f"Epoch {epoch} 저장 완료")


# ============================================================
# 8. 학습 종료
# ============================================================

print("\n[6/6] 학습 완료")

print("최종 저장 경로:", run_dir.resolve())
print("Last:", (weights_dir / "last.pt").resolve())
print("Best:", (weights_dir / "best.pt").resolve())
print("Log:", log_path.resolve())
print("최고 Validation Loss:", best_val_loss)

print("\nYOLOv8 Segmentation DirectML 학습 종료")
