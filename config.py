from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

PDF_DIR = ROOT / "input_pdf"
DATASET_DIR = ROOT / "train_data"
PAGES_DIR = DATASET_DIR / "pages_jpeg"
CROPS_DIR = DATASET_DIR / "crop_img"

DPI = 800
JPEG_QUALITY = 100

DET_MODEL = "PP-OCRv5_server_det"
MIN_SCORE = 0.45
PAD = 3