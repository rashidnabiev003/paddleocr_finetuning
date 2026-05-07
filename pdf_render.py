from pathlib import Path
import sys

import fitz
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[0]

PDF_DIR = ROOT / "input_pdf"
OUT_DIR = ROOT / "train_data" / "pages_jpeg"

DPI_START = 1000
DPI_FALLBACKS = [1000]
JPEG_QUALITY = 100


def ensure_dirs():
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def pixmap_to_image(pix: fitz.Pixmap) -> Image.Image:
    if pix.alpha:
        pix = fitz.Pixmap(fitz.csRGB, pix)

    mode = "RGB"
    size = (pix.width, pix.height)

    return Image.frombytes(mode, size, pix.samples)


def render_page(page: fitz.Page, out_path: Path) -> int:
    last_error = None

    for dpi in DPI_FALLBACKS:
        try:
            pix = page.get_pixmap(
                dpi=dpi,
                colorspace=fitz.csRGB,
                alpha=False,
                annots=True,
            )

            img = pixmap_to_image(pix)
            img.save(
                out_path,
                format="JPEG",
                quality=JPEG_QUALITY,
                subsampling=0,
                optimize=False,
                dpi=(dpi, dpi),
            )

            return dpi

        except Exception as e:
            last_error = e

    raise RuntimeError(f"Cannot render page {page.number}: {last_error}")


def render_pdf(pdf_path: Path):
    doc_name = pdf_path.stem

    with fitz.open(pdf_path) as doc:
        if doc.is_encrypted:
            raise RuntimeError(f"Encrypted PDF: {pdf_path}")

        if len(doc) == 0:
            raise RuntimeError(f"Empty PDF: {pdf_path}")

        for page_idx, page in enumerate(doc):
            out_path = OUT_DIR / f"{doc_name}_p{page_idx:03d}.jpg"
            used_dpi = render_page(page, out_path)
            print(f"[OK] {pdf_path.name} page={page_idx} dpi={used_dpi} -> {out_path}")


def main():
    ensure_dirs()

    pdfs = sorted(PDF_DIR.glob("*.pdf"))

    if not pdfs:
        print(f"No PDF files found in: {PDF_DIR}")
        sys.exit(1)

    failed = []

    for pdf_path in tqdm(pdfs, desc="render pdf"):
        try:
            render_pdf(pdf_path)
        except Exception as e:
            failed.append((pdf_path, str(e)))
            print(f"[FAIL] {pdf_path}: {e}")

    if failed:
        print("\nFailed PDFs:")
        for path, err in failed:
            print(f"- {path}: {err}")
        sys.exit(2)


if __name__ == "__main__":
    main()