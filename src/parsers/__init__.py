from .parser_factory import parse_document
from .pdf_parser import parse_pdf
from .docx_parser import parse_docx
from .doc_parser import parse_doc
from .txt_parser import parse_txt

__all__ = ["parse_pdf", "parse_docx", "parse_doc", "parse_txt", "parse_document"]
