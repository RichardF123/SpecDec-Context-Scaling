from pathlib import Path

from PIL import Image, ImageDraw
from pypdf import PdfReader


def main():
    root = Path(__file__).resolve().parents[1] / "output" / "pdf"
    pdf = root / "specdec_correctness_localization_report_2026-10-07.pdf"
    reader = PdfReader(pdf)
    texts = [page.extract_text() or "" for page in reader.pages]
    assert all(text.strip() for text in texts), "Blank text page"
    joined = "\n".join(texts)
    for marker in ("serial oracle", "B6_BF16_PASS", "39", "40", "CASE_INCOMPLETE"):
        assert marker in joined, marker
    previews = [root / f"correctness_rev2_preview-{index:02d}.png"
                for index in range(1, len(texts) + 1)]
    assert all(path.exists() for path in previews)
    for offset in range(0, len(previews), 12):
        selected = previews[offset:offset + 12]
        sheet = Image.new("RGB", (1200, 4 * 445), "#dfe5e8")
        draw = ImageDraw.Draw(sheet)
        for index, path in enumerate(selected):
            with Image.open(path) as page:
                page.thumbnail((385, 415))
                x, y = (index % 3) * 400, (index // 3) * 445
                sheet.paste(page, (x + 7, y + 22))
                draw.text((x + 7, y + 5), f"Page {offset + index + 1}", fill="black")
        sheet.save(root / f"correctness_rev2_contact_{offset // 12 + 1}.png")
    print(f"PDF text checks passed: {len(texts)} pages; contact sheets generated.")


if __name__ == "__main__":
    main()
