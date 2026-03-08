def extract_text(uploaded_file) -> str:
    """
    Extract plain text from an uploaded resume file.
    Supports PDF and DOCX formats.

    Args:
        uploaded_file: a Streamlit UploadedFile object

    Returns:
        Extracted text as a string
    """
    filename = uploaded_file.name

    if filename.endswith(".pdf"):
        return _extract_from_pdf(uploaded_file)

    elif filename.endswith(".docx"):
        return _extract_from_docx(uploaded_file)

    else:
        return ""


def _extract_from_pdf(uploaded_file) -> str:
    from pypdf import PdfReader

    reader = PdfReader(uploaded_file)
    pages = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text)
    return "\n".join(pages)


def _extract_from_docx(uploaded_file) -> str:
    import docx

    doc = docx.Document(uploaded_file)
    paragraphs = [para.text for para in doc.paragraphs if para.text.strip()]
    return "\n".join(paragraphs)