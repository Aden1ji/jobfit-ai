import re


def extract_skills(text: str, skills_path: str = "data/skills_list.txt") -> list[str]:
    """
    Scan resume text for known skills using whole-word matching.

    Args:
        text: raw resume text
        skills_path: path to the skills list file

    Returns:
        Sorted list of matched skill strings
    """
    skills = _load_skills(skills_path)
    normalized = _normalize(text)
    found = []
    for skill in skills:
        pattern = r"\b" + re.escape(skill) + r"\b"
        if re.search(pattern, normalized):
            found.append(skill)
    return sorted(set(found))


def _load_skills(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip().lower() for line in f if line.strip()]


def _normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s\+#]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text