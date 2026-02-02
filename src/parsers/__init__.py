from .parser_factory import parse_document
from .pdf_parser import parse_pdf
from .docx_parser import parse_docx
from .txt_parser import parse_txt

__all__ = ["parse_pdf", "parse_docx", "parse_txt", "parse_document"]
