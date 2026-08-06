"""
P0 improvement self-tests (run with the project's managed venv python + pytest).
Each test imports the code it needs locally so the file can be extended per task
without breaking collection of unrelated tests.

Run a subset:  pytest tests/test_p0_improvements.py -k pdf
"""
import glob
import os


def _first_text_pdf(root: str = "data/docs"):
    """Return a path to a PDF under root whose pypdf extraction is non-trivial."""
    from src.ingestion import _pdf_text_pypdf

    for path in sorted(glob.glob(os.path.join(root, "**", "*.pdf"), recursive=True)):
        try:
            if len((_pdf_text_pypdf(path) or "").strip()) > 100:
                return path
        except Exception:
            continue
    return None


# ----------------------------- P0-1: PDF extraction -----------------------------
def test_pdf_loader_nonempty():
    from src.ingestion import _load_pdf

    path = _first_text_pdf()
    assert path is not None, "no extractable PDF found in data/docs for testing"
    text = _load_pdf(path)
    assert isinstance(text, str) and len(text.strip()) > 0, f"fitz loader returned empty for {path}"


def test_pdf_loader_ge_pypdf():
    from src.ingestion import _load_pdf, _pdf_text_pypdf

    path = _first_text_pdf()
    assert path is not None
    fitz_len = len(_load_pdf(path).strip())
    pypdf_len = len(_pdf_text_pypdf(path).strip())
    # PyMuPDF should recover at least as much text as pypdf on a text-based PDF.
    assert fitz_len >= pypdf_len, (
        f"fitz extracted fewer chars ({fitz_len}) than pypdf ({pypdf_len}) for {path}"
    )
