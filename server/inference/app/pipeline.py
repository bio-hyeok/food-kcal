"""사진 한 장 → 음식 상자 목록 (POST /detect 와 평가 스크립트가 같이 쓴다).

1. 음식 검출 모델(DETECT_MODEL_DIR)로 음식 상자를 찾는다.
2. 그릇 검출 모델(DETECT_BOWL_MODEL_DIR, 있으면)로 그릇 상자를 찾는다. 실제 상차림처럼 그릇이 빽빽한 사진에서
   음식 검출 모델은 상 전체를 상자 하나로 잡는 경우가 많아서 보탠다 (실제 상차림 20장: 아는 음식 찾음 14% → 96%).
3. 합치기 (merge_boxes)
   - 사진 넓이의 bowl_max_area 이상인 그릇 상자는 버린다 (상·쟁반 전체)
   - 사진 넓이의 table_min_area 이상이고 그릇 상자 가운데가 table_min_bowls 개 이상 들어 있는 음식 상자는 버린다
     (상 전체를 음식 하나로 본 상자). 2개·넓이 조건 없음으로 하면 한 음식 test 사진의 5% 에서 진짜 음식 상자가 버려졌다
   - 음식 상자를 먼저 두고, 이미 둔 상자와 겹치는(IoU ≥ dup_iou) 그릇 상자는 뺀다
4. 상자마다 잘라 분류한다. 그릇 상자는 이름을 모르는 반찬(김치·나물 등)도 많이 잡아, 1순위를 미리 고르면
   틀린 음식이 합계에 들어가기 쉽다. 그래서 그릇 상자는 기본으로 status=FAILED (후보만 보여 주고 사용자가 고른다).
"""
from dataclasses import dataclass

import numpy as np
from PIL import Image

from .classifier import Candidate, Classifier
from .detector import Box, Detector, crop_box, iou


@dataclass
class Item:
    box: Box
    source: str            # "food" 또는 "bowl"
    candidates: list[Candidate]
    status: str            # SUCCESS / FAILED


def _xyxy(b: Box) -> np.ndarray:
    return np.array([b.x0, b.y0, b.x1, b.y1], dtype=np.float64)


def merge_boxes(food: list[Box], bowls: list[Box], w: int, h: int, *, bowl_max_area: float = 0.4,
                dup_iou: float = 0.4, max_items: int = 15, table_min_area: float = 0.4,
                table_min_bowls: int = 3) -> list[tuple[Box, str]]:
    bowls = [b for b in bowls if (b.x1 - b.x0) * (b.y1 - b.y0) < bowl_max_area * w * h]

    def centers_inside(f: Box) -> int:
        return sum(f.x0 <= (b.x0 + b.x1) / 2 <= f.x1 and f.y0 <= (b.y0 + b.y1) / 2 <= f.y1 for b in bowls)

    def is_table(f: Box) -> bool:
        return (f.x1 - f.x0) * (f.y1 - f.y0) >= table_min_area * w * h and centers_inside(f) >= table_min_bowls

    kept = [(b, "food") for b in food if not is_table(b)]
    for b in bowls:   # 신뢰도 높은 순
        if kept and iou(_xyxy(b), np.stack([_xyxy(k) for k, _ in kept])).max() >= dup_iou:
            continue
        kept.append((b, "bowl"))
    return kept[:max_items]


def detect_items(im: Image.Image, s, clf: Classifier, det: Detector, bowl: Detector | None = None) -> list[Item]:
    w, h = im.size
    food = det.detect(im, s.detect_min_confidence, s.detect_iou, s.detect_max_items)
    bowls = bowl.detect(im, s.detect_bowl_min_confidence, s.detect_iou, 50) if bowl else []
    items = []
    for b, src in merge_boxes(food, bowls, w, h, max_items=s.detect_max_items):
        cands = clf.predict(crop_box(im, b), s.top_k)
        sure = cands[0].confidence >= s.min_confidence and (src == "food" or s.detect_bowl_auto_select)
        items.append(Item(b, src, cands, "SUCCESS" if sure else "FAILED"))
    return items
