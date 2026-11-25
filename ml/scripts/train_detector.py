"""음식 위치 찾기(검출) 모델 학습. 데이터는 prepare_detect_data.py 결과.

종류는 가리지 않고 "food" 한 클래스만 찾는다. 찾은 상자를 잘라 분류 모델(v3)에 넣는다.
Ultralytics YOLO 를 쓴다 (AGPL-3.0. 과제·연구용. 상업 서비스로 쓰려면 라이선스를 다시 검토한다).

    python scripts/train_detector.py --data C:\\food-data\\detect\\data.yaml --project C:\\food-data\\detect_runs
"""
import argparse

from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--project", required=True)
    ap.add_argument("--name", default="yolo26n_food")
    ap.add_argument("--model", default="yolo26n.pt", help="COCO 사전학습 가중치")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()

    model = YOLO(a.model)
    model.train(data=a.data, project=a.project, name=a.name, epochs=a.epochs, imgsz=a.imgsz,
                batch=a.batch, workers=a.workers, seed=42, exist_ok=True, plots=True)


if __name__ == "__main__":   # Windows 에서 DataLoader worker 를 쓰려면 필요하다
    main()
