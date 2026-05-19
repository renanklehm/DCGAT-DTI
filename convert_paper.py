from pathlib import Path

from markitdown import MarkItDown


def main() -> None:
    project_root = Path(__file__).resolve().parent
    source = project_root / "paper.pdf"
    target = project_root / "paper.md"

    result = MarkItDown().convert(source)
    target.write_text(result.text_content, encoding="utf-8")


if __name__ == "__main__":
    main()
