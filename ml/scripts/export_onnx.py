"""분류 모델을 ONNX 로 변환하고 PyTorch 결과와 같은지 확인한다 (위키 전체 흐름 문서 6.9).

입력: <model-dir>/classifier_best.pt, idx_to_food_id.json, train_config.json
출력: <model-dir>/classifier.onnx, model_meta.json (추론 서버가 전처리를 똑같이 하기 위한 값)

사용 (로컬 Windows):
    python scripts\\export_onnx.py --model-dir C:\\food-data\\models_v2 --data C:\\food-data\\processed_full

확인하는 것
1. 같은 test 사진에서 PyTorch 와 ONNX Runtime 의 확률 차이(최대값)와 1순위 일치율
2. ONNX Runtime(CPU) 으로 test 전체를 다시 평가한 top-1 / top-5
3. CPU 에서 사진 1장 추론 시간 (전처리 제외)
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--data", required=True, help="test 폴더가 있는 곳 (확인용)")
    ap.add_argument("--opset", type=int, default=18)
    ap.add_argument("--check", type=int, default=256, help="PyTorch 와 비교할 test 사진 수")
    ap.add_argument("--skip-full-eval", action="store_true")
    a = ap.parse_args()
    md = Path(a.model_dir)

    cfg = json.loads((md / "train_config.json").read_text(encoding="utf-8")) if (md / "train_config.json").exists() else {}
    size = cfg.get("img_size", 224)
    resize = round(size * 256 / 224)
    idx_to_food_id = {int(k): v for k, v in json.loads((md / "idx_to_food_id.json").read_text(encoding="utf-8")).items()}

    model = models.efficientnet_b0()
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(idx_to_food_id))
    ckpt = md / "classifier_best.pt"
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model.eval()

    # 1) 변환. 배치 크기는 바꿀 수 있게 둔다.
    onnx_path = md / "classifier.onnx"
    dummy = torch.randn(1, 3, size, size)
    torch.onnx.export(
        model, (dummy,), str(onnx_path),
        input_names=["image"], output_names=["logits"],
        dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=a.opset, dynamo=False,
    )
    meta = {
        "architecture": "efficientnet_b0",
        "onnx": onnx_path.name,
        "input": {"name": "image", "layout": "NCHW", "dtype": "float32", "size": size,
                  "resize_short_side": resize, "center_crop": size, "color": "RGB",
                  "scale": "0~1 (픽셀/255)", "mean": MEAN, "std": STD},
        "output": {"name": "logits", "num_classes": len(idx_to_food_id), "note": "softmax 를 적용해 확률로 쓴다"},
        "idx_to_food_id": {str(k): v for k, v in sorted(idx_to_food_id.items())},
        "source_checkpoint": ckpt.name,
        "source_sha1": hashlib.sha1(ckpt.read_bytes()).hexdigest(),
        "train_config": cfg,
    }
    (md / "model_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"변환 완료: {onnx_path} ({onnx_path.stat().st_size / 1e6:.1f}MB), 입력 {size}x{size}")

    tf = transforms.Compose([
        transforms.Resize(resize), transforms.CenterCrop(size),
        transforms.ToTensor(), transforms.Normalize(MEAN, STD),
    ])
    ds = datasets.ImageFolder(f"{a.data}/test", tf)
    ds_idx_to_food_id = {idx: int(name) for name, idx in ds.class_to_idx.items()}
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    # 2) PyTorch 와 비교 (CPU, 같은 입력)
    step = max(1, len(ds) // a.check)
    sub = Subset(ds, list(range(0, len(ds), step))[: a.check])
    max_diff, same = 0.0, 0
    for x, _ in DataLoader(sub, batch_size=64):
        with torch.no_grad():
            p_t = torch.softmax(model(x), 1).numpy()
        p_o = torch.softmax(torch.from_numpy(sess.run(None, {"image": x.numpy()})[0]), 1).numpy()
        max_diff = max(max_diff, float(np.abs(p_t - p_o).max()))
        same += int((p_t.argmax(1) == p_o.argmax(1)).sum())
    print(f"PyTorch 비교 {len(sub)}장: 확률 최대 차이 {max_diff:.2e}, 1순위 일치 {same}/{len(sub)}")
    if same != len(sub) or max_diff > 1e-3:
        raise SystemExit("ONNX 결과가 PyTorch 와 다릅니다. 변환을 확인하세요.")

    # 3) 1장 추론 시간 (CPU, 전처리 제외)
    one = ds[0][0].unsqueeze(0).numpy()
    for _ in range(5):
        sess.run(None, {"image": one})
    t0 = time.perf_counter()
    for _ in range(50):
        sess.run(None, {"image": one})
    ms = (time.perf_counter() - t0) / 50 * 1000
    print(f"CPU 1장 추론: 평균 {ms:.1f}ms")
    meta["cpu_ms_per_image_local"] = round(ms, 1)

    # 4) ONNX Runtime 으로 test 전체 평가
    if not a.skip_full_eval:
        n = top1 = top5 = 0
        for x, y in DataLoader(ds, batch_size=64, num_workers=4):
            logits = sess.run(None, {"image": x.numpy()})[0]
            top = np.argsort(-logits, axis=1)[:, :5]
            for t5, yi in zip(top, y.tolist()):
                true_id = ds_idx_to_food_id[yi]
                ids = [idx_to_food_id[int(i)] for i in t5]
                n += 1
                top1 += ids[0] == true_id
                top5 += true_id in ids
        meta["onnx_test"] = {"images": n, "top1": round(top1 / n, 4), "top5": round(top5 / n, 4)}
        print(f"ONNX Runtime test {n}장: top-1 {top1 / n:.2%}, top-5 {top5 / n:.2%}")

    (md / "model_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"메타 정보: {md / 'model_meta.json'}")


if __name__ == "__main__":
    main()
