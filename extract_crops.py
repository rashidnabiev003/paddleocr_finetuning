import csv
import json
from pathlib import Path

import cv2
import numpy as np
from paddleocr import PaddleOCR
from tqdm import tqdm


ROOT = Path(__file__).resolve().parent

DATASET_DIR = ROOT / "train_data"
PAGES_DIR = DATASET_DIR / "pages_jpeg"
CROPS_DIR = DATASET_DIR / "crop_images"

PAD = 3
MIN_SCORE = 0.3


def imread_unicode(path: Path):
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def imwrite_unicode(path: Path, img) -> bool:
    ok, buf = cv2.imencode(path.suffix, img)
    if not ok:
        return False
    buf.tofile(str(path))
    return True


def result_payload(res):
    data = res.json if hasattr(res, "json") else res
    return data.get("res", data)


def order_points(poly):
    pts = np.asarray(poly, dtype=np.float32)

    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1)

    return np.array(
        [
            pts[np.argmin(s)],
            pts[np.argmin(d)],
            pts[np.argmax(s)],
            pts[np.argmax(d)],
        ],
        dtype=np.float32,
    )


def crop_poly(img, poly):
    pts = order_points(poly)

    w = int(max(
        np.linalg.norm(pts[1] - pts[0]),
        np.linalg.norm(pts[2] - pts[3]),
    ))

    h = int(max(
        np.linalg.norm(pts[3] - pts[0]),
        np.linalg.norm(pts[2] - pts[1]),
    ))

    if w < 4 or h < 4:
        return None

    dst = np.float32([
        [0, 0],
        [w - 1, 0],
        [w - 1, h - 1],
        [0, h - 1],
    ])

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

    return cv2.copyMakeBorder(
        crop,
        PAD,
        PAD,
        PAD,
        PAD,
        cv2.BORDER_REPLICATE,
    )


def sort_boxes(polys, scores):
    items = []

    for poly, score in zip(polys, scores):
        p = np.asarray(poly)
        x = p[:, 0].mean()
        y = p[:, 1].mean()
        items.append((round(y / 30) * 30, x, poly, score))

    return [(poly, score) for _, _, poly, score in sorted(items)]


def find_pages():
    if not PAGES_DIR.exists():
        raise FileNotFoundError(PAGES_DIR)

    pages = sorted(
        p for p in PAGES_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )

    if not pages:
        raise FileNotFoundError(f"No images found in: {PAGES_DIR}")

    return pages


def main():
    CROPS_DIR.mkdir(parents=True, exist_ok=True)

    ocr = PaddleOCR(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,

        # параметры детектора из рабочего pipeline-режима
        text_det_limit_side_len=736,
        text_det_limit_type="min",
        text_det_thresh=0.3,
        text_det_box_thresh=0.6,
        text_det_unclip_ratio=1.5,

        # чтобы recognition ничего не отфильтровывал
        text_rec_score_thresh=0.0,
    )

    manifest = DATASET_DIR / "manifest.tsv"
    pages = find_pages()

    with manifest.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow([
            "crop_path",
            "label",
            "page_image",
            "box",
            "dt_score",
            "rec_text",
            "rec_score",
            "poly",
        ])

        for page_path in tqdm(pages, desc="extract crops"):
            img = imread_unicode(page_path)

            if img is None:
                print(f"[SKIP] cannot read image: {page_path}")
                continue

            try:
                h, w = img.shape[:2]
                max_dim = 3500
                if h > max_dim or w > max_dim:
                    scale = max_dim / max(h, w)
                    img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

                result = ocr.predict(img)
            except cv2.error as e:
                tqdm.write(f"[WARN] Skipping {page_path.name} : {e}")
                continue
            except Exception as e:
                tqdm.write(f"[WARN] Skipping {page_path.name} : {e}")
                continue

            for res in result:
                payload = result_payload(res)

                print("[DEBUG]", page_path.name, payload.get("text_det_params"))

                polys = payload.get("dt_polys", [])
                scores = payload.get("dt_scores", [1.0] * len(polys))

                rec_texts = payload.get("rec_texts", [])
                rec_scores = payload.get("rec_scores", [])

                print(
                    f"[BOXES] {page_path.name}: "
                    f"dt_polys={len(polys)}, "
                    f"rec_texts={len(rec_texts)}, "
                    f"rec_scores={len(rec_scores)}"
                )

                for idx, (poly, score) in enumerate(sort_boxes(polys, scores)):
                    score = float(score)

                    if score < MIN_SCORE:
                        continue

                    crop = crop_poly(img, poly)

                    if crop is None:
                        continue

                    crop_path = CROPS_DIR / f"{page_path.stem}_b{idx:04d}.png"

                    if not imwrite_unicode(crop_path, crop):
                        print(f"[SKIP] cannot write crop: {crop_path}")
                        continue

                    rel = crop_path.relative_to(DATASET_DIR).as_posix()

                    rec_text = rec_texts[idx] if idx < len(rec_texts) else ""
                    rec_score = float(rec_scores[idx]) if idx < len(rec_scores) else ""

                    writer.writerow([
                        rel,
                        "",
                        page_path.name,
                        idx,
                        score,
                        rec_text,
                        rec_score,
                        json.dumps(np.asarray(poly).tolist(), ensure_ascii=False),
                    ])


if __name__ == "__main__":
    main()