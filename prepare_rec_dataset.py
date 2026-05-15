from pathlib import Path
import argparse
import csv
import random
import unicodedata


ROOT = Path(__file__).resolve().parent
TRAIN_DATA = ROOT / "train_data"
LABEL_CSV = TRAIN_DATA / "label.csv"

TRAIN_LIST = TRAIN_DATA / "train_list.txt"
VAL_LIST = TRAIN_DATA / "val_list.txt"

REC_GT_TRAIN = TRAIN_DATA / "rec_gt_train.txt"
REC_GT_VAL = TRAIN_DATA / "rec_gt_val.txt"

DICT_PATH = TRAIN_DATA / "dict_passport.txt"
SKIPPED_PATH = TRAIN_DATA / "skipped_rows.txt"


BASE_CHARS = (
    "0123456789"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "abcdefghijklmnopqrstuvwxyz"
    "АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ"
    "абвгдеёжзийклмнопрстуфхцчшщъыьэюя"
    "<>/-.,:;№()[]"
)


def clean_label(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\ufeff", "")
    text = text.replace("\xa0", " ")
    text = " ".join(text.split())
    return text.strip()


def sniff_dialect(path: Path):
    sample = path.read_text(encoding="utf-8-sig", errors="ignore")[:4096]
    try:
        return csv.Sniffer().sniff(sample, delimiters=",;\t")
    except csv.Error:
        return csv.excel


def read_label_csv():
    rows = []
    skipped = []

    dialect = sniff_dialect(LABEL_CSV)

    with LABEL_CSV.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f, dialect)

        for line_no, row in enumerate(reader, start=1):
            if len(row) < 2:
                skipped.append(f"{line_no}: less than 2 columns")
                continue

            crop_path = row[0].strip().replace("\\", "/")
            label = clean_label(row[1])

            if line_no == 1 and crop_path.lower() in {"crop_path", "path", "image"}:
                continue

            if not crop_path:
                skipped.append(f"{line_no}: empty crop_path")
                continue

            if not label:
                skipped.append(f"{line_no}: empty label")
                continue

            if "\t" in label or "\n" in label or "\r" in label:
                skipped.append(f"{line_no}: bad label control char")
                continue

            image_path = TRAIN_DATA / crop_path

            if not image_path.exists():
                skipped.append(f"{line_no}: image not found: {crop_path}")
                continue

            rows.append((crop_path, label))

    return rows, skipped


def page_key(crop_path: str) -> str:
    stem = Path(crop_path).stem
    return stem.split("_b")[0] if "_b" in stem else stem


def split_rows(rows, val_ratio: float, seed: int):
    groups = {}

    for path, label in rows:
        groups.setdefault(page_key(path), []).append((path, label))

    group_items = list(groups.items())
    random.Random(seed).shuffle(group_items)

    val_size = max(1, int(len(group_items) * val_ratio))
    val_keys = {key for key, _ in group_items[:val_size]}

    train, val = [], []

    for key, items in group_items:
        if key in val_keys:
            val.extend(items)
        else:
            train.extend(items)

    if not train and val:
        train.append(val.pop())

    return train, val


def write_list(path: Path, rows):
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for crop_path, label in rows:
            f.write(f"{crop_path}\t{label}\n")


def write_dict(rows):
    chars = set(BASE_CHARS)

    for _, label in rows:
        for ch in label:
            if ch != " ":
                chars.add(ch)

    with DICT_PATH.open("w", encoding="utf-8", newline="\n") as f:
        for ch in sorted(chars):
            f.write(ch + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows, skipped = read_label_csv()

    if not rows:
        raise RuntimeError("No valid labeled rows found")

    train, val = split_rows(rows, args.val_ratio, args.seed)

    write_list(TRAIN_LIST, train)
    write_list(VAL_LIST, val)

    write_list(REC_GT_TRAIN, train)
    write_list(REC_GT_VAL, val)

    write_dict(rows)

    SKIPPED_PATH.write_text("\n".join(skipped), encoding="utf-8")

    print(f"valid total: {len(rows)}")
    print(f"train:       {len(train)} -> {TRAIN_LIST}")
    print(f"val:         {len(val)} -> {VAL_LIST}")
    print(f"dict:        {DICT_PATH}")
    print(f"skipped:     {len(skipped)} -> {SKIPPED_PATH}")


if __name__ == "__main__":
    main()