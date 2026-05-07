import csv
import json

import cv2
import numpy as np
from paddleocr import TextDetection
from tqdm import tqdm

from config import PAGES_DIR, CROPS_DIR, DATASET_DIR, DET_MODEL, MIN_SCORE, PAD


def result_payload(res):
    data = res.json if hasattr(res, "json") else res
    return data.get("res", data)


def order_points(poly):
    pts = np.asarray(poly, dtype=np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1)

    return np.array([
        pts[np.argmin(s)],
        pts[np.argmin(d)],
        pts[np.argmax(s)],
        pts[np.argmax(d)],
    ], dtype=np.float32)


def crop_poly(img, poly):
    pts = order_points(poly)

    w = int(max(np.linalg.norm(pts[1] - pts[0]), np.linalg.norm(pts[2] - pts[3])))
    h = int(max(np.linalg.norm(pts[3] - pts[0]), np.linalg.norm(pts[2] - pts[1])))

    if w < 4 or h < 4:
        return None

    dst = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    mat = cv2.getPerspectiveTransform(pts, dst)

    crop = cv2.warpPerspective(
        img,
        mat,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )

    if crop.shape[0] > crop.shape[1] * 1.5:
        crop = cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)

    return cv2.copyMakeBorder(crop, PAD, PAD, PAD, PAD, cv2.BORDER_REPLICATE)


def sort_boxes(polys, scores):
    items = []

    for poly, score in zip(polys, scores):
        p = np.asarray(poly)
        x = p[:, 0].mean()
        y = p[:, 1].mean()
        items.append((round(y / 30) * 30, x, poly, score))

    return [(poly, score) for _, _, poly, score in sorted(items)]


def main():
    CROPS_DIR.mkdir(parents=True, exist_ok=True)

    detector = TextDetection(model_name=DET_MODEL)
    manifest = DATASET_DIR / "manifest.tsv"

    with manifest.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["crop_path", "label", "doc", "page", "box", "score", "poly"])

        pages = sorted(PAGES_DIR.glob("*/*.jpg"))

        for page_path in tqdm(pages, desc="extract crops"):
            img = cv2.imread(str(page_path))
            doc_name = page_path.parent.name
            page_name = page_path.stem

            out_dir = CROPS_DIR / doc_name / page_name
            out_dir.mkdir(parents=True, exist_ok=True)

            result = detector.predict(str(page_path), batch_size=1)

            for res in result:
                payload = result_payload(res)
                polys = payload.get("dt_polys", [])
                scores = payload.get("dt_scores", [1.0] * len(polys))

                for idx, (poly, score) in enumerate(sort_boxes(polys, scores)):
                    score = float(score)

                    if score < MIN_SCORE:
                        continue

                    crop = crop_poly(img, poly)

                    if crop is None:
                        continue

                    crop_path = out_dir / f"b{idx:04d}.png"
                    cv2.imwrite(str(crop_path), crop)

                    rel = crop_path.relative_to(DATASET_DIR).as_posix()

                    writer.writerow([
                        rel,
                        "",
                        doc_name,
                        page_name,
                        idx,
                        score,
                        json.dumps(np.asarray(poly).tolist(), ensure_ascii=False),
                    ])


if __name__ == "__main__":
    main()