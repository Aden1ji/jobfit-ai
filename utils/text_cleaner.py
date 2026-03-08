import re


def clean_text(text: str) -> str:
    """
    Normalize text for skill matching and processing.

    Steps:
      1. Lowercase everything
      2. Remove non-alphanumeric characters (keeps spaces, + and #)
      3. Collapse multiple spaces into one
    """
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s\+#]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text