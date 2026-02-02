"""
Verify and compare text extraction across formats.
Run: python scripts/verify_extraction.py path/to/document

Compares extraction from PDF vs DOCX vs TXT when same base file exists in multiple formats.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.parsers import parse_document


def main():
    if len(sys.argv) < 2:
        base = Path(__file__).parent.parent / "test2"
        paths = [base.with_suffix(".pdf"), base.with_suffix(".DOCX"), base.with_suffix(".docx"), base.with_suffix(".txt")]
        for p in paths:
            if p.exists():
                print(f"Usage: python {sys.argv[0]} path/to/document")
                print(f"\nTrying: {p}")
                _verify(p)
                return
        print("Usage: python verify_extraction.py <path_to_document>")
        print("Example: python verify_extraction.py test2.pdf")
        sys.exit(1)

    path = Path(sys.argv[1])
    if not path.exists():
        print(f"File not found: {path}")
        sys.exit(1)

    _verify(path)


def _verify(path: Path):
    stem = path.stem
    parent = path.parent
    formats_to_try = [
        (parent / f"{stem}.pdf", "PDF"),
        (parent / f"{stem}.docx", "DOCX"),
        (parent / f"{stem}.DOCX", "DOCX"),
        (parent / f"{stem}.txt", "TXT"),
    ]

    results = {}
    for p, name in formats_to_try:
        if p.exists() and p.suffix.lower() in (".pdf", ".docx", ".txt"):
            try:
                parsed = parse_document(p)
                text = parsed.get("text", "")
                words = len(text.split())
                chars = len(text)
                results[str(p)] = {"format": name, "chars": chars, "words": words, "preview": text[:300]}
            except Exception as e:
                results[str(p)] = {"format": name, "error": str(e)}

    print("\n" + "=" * 60)
    print("EXTRACTION VERIFICATION")
    print("=" * 60)

    for filepath, r in results.items():
        print(f"\n--- {Path(filepath).name} ({r.get('format', '?')}) ---")
        if "error" in r:
            print(f"  ERROR: {r['error']}")
        else:
            print(f"  Characters: {r['chars']:,}")
            print(f"  Words: {r['words']:,}")
            print(f"  Preview: {repr(r['preview'][:150])}...")

    if len(results) >= 2 and "error" not in str(results):
        chars_list = [r.get("chars", 0) for r in results.values() if "chars" in r]
        if chars_list:
            diff_pct = (max(chars_list) - min(chars_list)) / max(chars_list) * 100 if max(chars_list) else 0
            print(f"\n--- Comparison ---")
            print(f"  Character difference between formats: {diff_pct:.1f}%")
            if diff_pct > 20:
                print("  WARNING: Large difference - extraction may vary significantly by format.")


if __name__ == "__main__":
    main()
