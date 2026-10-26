"""음식 분류 학습 (EfficientNet-B0 파인튜닝). 위키 전체 흐름 문서 6.7 의 초안을 옮긴 것이다.

입력: prepare_classifier_data.py 가 만든 <data>/train, <data>/val (폴더 이름 = food_id)
출력: <model-dir>/idx_to_food_id.json, classifier_best.pt

사용 (Colab, GPU 런타임):
    !python ml/scripts/train_classifier.py --data /content/processed --model-dir /content/models

실행 확인 범위: 위키 기준으로는 무작위 이미지 48장으로 CPU에서만 끝까지 돌려 봤다.
실제 음식 데이터에서의 성능과 속도는 확인하지 못했다.
"""
import argparse
import json
import os
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed")
    ap.add_argument("--model-dir", default="models")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--head-epochs", type=int, default=3)
    ap.add_argument("--full-epochs", type=int, default=10)
    a = ap.parse_args()

    # Windows 에서는 0 이 안전하다. Linux/WSL/Colab 에서는 4.
    num_workers = 0 if sys.platform == "win32" else 4
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(a.model_dir, exist_ok=True)
    print("device:", device)

    norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    tf_train = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2),
        transforms.ToTensor(),
        norm,
    ])
    tf_eval = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        norm,
    ])

    train_ds = datasets.ImageFolder(f"{a.data}/train", tf_train)
    val_ds = datasets.ImageFolder(f"{a.data}/val", tf_eval)
    train_dl = DataLoader(train_ds, batch_size=a.batch, shuffle=True, num_workers=num_workers)
    val_dl = DataLoader(val_ds, batch_size=a.batch, num_workers=num_workers)
    print(f"클래스 {len(train_ds.classes)}개, train {len(train_ds)}장, val {len(val_ds)}장")

    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(train_ds.classes))
    model.to(device)
    criterion = nn.CrossEntropyLoss()

    def run_epoch(loader, optimizer=None):
        training = optimizer is not None
        model.train(training)
        total, correct, loss_sum = 0, 0, 0.0
        with torch.set_grad_enabled(training):
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                loss = criterion(out, y)
                if training:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                loss_sum += loss.item() * x.size(0)
                correct += (out.argmax(1) == y).sum().item()
                total += x.size(0)
        return round(loss_sum / total, 4), round(correct / total, 4)

    # 추론할 때 "클래스 번호 → food_id"가 필요하다. 폴더 이름은 문자열 순서라 "100"이 "12"보다 앞선다.
    idx_to_food_id = {idx: int(name) for name, idx in train_ds.class_to_idx.items()}
    with open(f"{a.model_dir}/idx_to_food_id.json", "w", encoding="utf-8") as f:
        json.dump(idx_to_food_id, f)

    # 1단계: 분류 헤드만 학습
    for p in model.features.parameters():
        p.requires_grad = False
    opt = torch.optim.AdamW(model.classifier.parameters(), lr=1e-3)
    for epoch in range(a.head_epochs):
        tr = run_epoch(train_dl, opt)
        va = run_epoch(val_dl)
        print(f"[head] epoch {epoch} train(loss, acc)={tr} val(loss, acc)={va}")

    # 2단계: 전체를 낮은 학습률로 파인튜닝
    for p in model.parameters():
        p.requires_grad = True
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    best = 0.0
    for epoch in range(a.full_epochs):
        tr = run_epoch(train_dl, opt)
        va = run_epoch(val_dl)
        print(f"[full] epoch {epoch} train(loss, acc)={tr} val(loss, acc)={va}")
        if va[1] >= best:
            best = va[1]
            torch.save(model.state_dict(), f"{a.model_dir}/classifier_best.pt")
    print("best val acc:", best)


if __name__ == "__main__":
    main()
