"""Legacy .doc (Word 97–2003) parser via conversion to .docx or plain text."""

import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from .docx_parser import parse_docx


def _find_soffice() -> str | None:
    """Locate LibreOffice soffice executable (Windows and Unix)."""
    # 1. Explicit path (for Streamlit Cloud or custom installs)
    env_path = os.environ.get("SOFFICE_PATH") or os.environ.get("LIBREOFFICE_PATH")
    if env_path:
        p = Path(env_path).resolve()
        if p.is_file():
            return str(p)
        if p.is_dir():
            exe = p / "soffice.exe" if os.name == "nt" else p / "soffice"
            if exe.exists():
                return str(exe)

    # 2. PATH
    found = shutil.which("soffice") or shutil.which("soffice.exe")
    if found:
        return found

    # 3. Windows: Program Files and versioned installs
    if os.name == "nt":
        candidates = [
            Path(r"C:\Program Files\LibreOffice\program\soffice.exe"),
            Path(r"C:\Program Files (x86)\LibreOffice\program\soffice.exe"),
        ]
        for base in [Path(r"C:\Program Files"), Path(r"C:\Program Files (x86)")]:
            if base.exists():
                try:
                    for sub in base.iterdir():
                        if sub.is_dir() and "LibreOffice" in sub.name:
                            exe = sub / "program" / "soffice.exe"
                            if exe.exists():
                                return str(exe)
                except OSError:
                    pass
        for p in candidates:
            if p.exists():
                return str(p)
        # User install (e.g. from LibreOffice installer "current user")
        local = os.environ.get("LOCALAPPDATA", "")
        if local:
            lo_base = Path(local) / "Programs" / "LibreOffice" / "program"
            if (lo_base / "soffice.exe").exists():
                return str(lo_base / "soffice.exe")
    return None


def _convert_doc_to_docx_with_libreoffice(doc_path: Path, out_dir: Path) -> Path | None:
    """Convert .doc to .docx using LibreOffice headless. Returns path to .docx or None."""
    soffice = _find_soffice()
    if not soffice:
        return None
    doc_path = doc_path.resolve()
    out_dir = out_dir.resolve()
    # Copy to a simple name in out_dir to avoid path-with-spaces or encoding issues
    simple_doc = out_dir / "input.doc"
    try:
        shutil.copy2(doc_path, simple_doc)
    except OSError:
        simple_doc = doc_path
    # Dedicated user profile (avoids lock/cache when multiple instances or no GUI)
    profile_dir = out_dir / "lo_profile"
    profile_dir.mkdir(exist_ok=True)
    profile_uri = profile_dir.as_uri()
    cmd = [
        soffice,
        "-env:UserInstallation=" + profile_uri,
        "--headless",
        "--invisible",
        "--nologo",
        "--nofirststartwizard",
        "--convert-to", "docx",
        "--outdir", str(out_dir),
        str(simple_doc),
    ]
    try:
        subprocess.run(
            cmd,
            capture_output=True,
            timeout=90,
            text=True,
            cwd=str(out_dir),
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    # Output: input.docx when we used simple_doc, else stem of original
    docx_path = out_dir / (simple_doc.stem + ".docx")
    if docx_path.exists():
        return docx_path
    for f in out_dir.iterdir():
        if f.suffix.lower() == ".docx" and f.is_file():
            return f
    return None


def _convert_doc_to_text_with_pandoc(doc_path: Path) -> str | None:
    """Convert .doc to plain text using pypandoc (requires pandoc binary). Returns text or None."""
    try:
        import pypandoc
    except ImportError:
        return None
    try:
        text = pypandoc.convert_file(str(doc_path), "plain", extra_args=["--wrap=none"])
        return text.strip() if text else None
    except Exception:
        return None


def parse_doc(file_path: str | Path) -> dict[str, Any]:
    """
    Extract text and structure from a legacy .doc file.
    Converts to .docx via LibreOffice (if available) and uses the DOCX parser,
    or falls back to pandoc for plain-text extraction.

    Args:
        file_path: Path to the .doc file.

    Returns:
        Dict with keys: text, structure, metadata.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the format is unsupported or conversion tools are missing.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"DOC file not found: {path}")

    path = path.resolve()
    suffix = path.suffix.lower()
    if suffix != ".doc":
        raise ValueError(f"Expected .doc file, got {suffix}")

    # 1. Try LibreOffice: .doc -> .docx, then parse with docx parser
    soffice_found = _find_soffice() is not None
    with tempfile.TemporaryDirectory(prefix="pyxon_doc_") as tmpdir:
        out_dir = Path(tmpdir)
        docx_path = _convert_doc_to_docx_with_libreoffice(path, out_dir)
        if docx_path is not None:
            return parse_docx(docx_path)

    # 2. Fallback: pandoc .doc -> plain text
    text = _convert_doc_to_text_with_pandoc(path)
    if text:
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        structure = [{"type": "paragraph", "level": 0, "text": p} for p in paragraphs]
        metadata = {
            "source": str(path),
            "filename": path.name,
            "format": "doc",
            "paragraph_count": len(paragraphs),
        }
        return {
            "text": text,
            "structure": structure,
            "metadata": metadata,
        }

    if soffice_found:
        raise ValueError(
            "LibreOffice was found but .doc conversion failed. Try opening the file in LibreOffice and saving as .docx, or use a file without special characters in the path."
        )
    raise ValueError(
        "Could not read .doc file. "
        "If you're on Streamlit Cloud, .doc is not supported—please upload .docx. "
        "Locally: install LibreOffice and ensure soffice.exe is on PATH, or set SOFFICE_PATH to its path (e.g. C:\\Program Files\\LibreOffice\\program\\soffice.exe), or save the file as .docx."
    )
