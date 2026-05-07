import fitz
from PIL import Image
from tqdm import tqdm

from config import PDF_DIR, PAGES_DIR, DPI, JPEG_QUALITY


def pixmap_to_pil(pix):
    mode = "RGB" if pix.alpha == 0 else "RGBA"
    return Image.frombytes(mode, [pix.width, pix.height], pix.samples).convert("RGB")


def render_pdf(pdf_path):
    doc_dir = PAGES_DIR / pdf_path.stem
    doc_dir.mkdir(parents=True, exist_ok=True)

    with fitz.open(pdf_path) as doc:
        for page_idx, page in enumerate(doc):
            out = doc_dir / f"p{page_idx:03d}.jpg"
            pix = page.get_pixmap(dpi=DPI, alpha=False)
            img = pixmap_to_pil(pix)
            img.save(out, "JPEG", quality=JPEG_QUALITY, subsampling=0, dpi=(DPI, DPI))


def main():
    PAGES_DIR.mkdir(parents=True, exist_ok=True)

    for pdf in tqdm(sorted(PDF_DIR.glob("*.pdf")), desc="render pdf"):
        render_pdf(pdf)


if __name__ == "__main__":
    main()