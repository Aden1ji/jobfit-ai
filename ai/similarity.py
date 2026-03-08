import math


def cosine_similarity(resume_skills: list[str], job_skills: list[str]) -> float:
    """
    Computes cosine similarity between two skill lists.
    Both lists are converted into binary vectors over a shared vocabulary.
    """
    all_skills = list(set(resume_skills + job_skills))

    resume_vector = [1 if skill in resume_skills else 0 for skill in all_skills]
    job_vector = [1 if skill in job_skills else 0 for skill in all_skills]

    dot_product = sum(a * b for a, b in zip(resume_vector, job_vector))
    mag1 = math.sqrt(sum(a * a for a in resume_vector))
    mag2 = math.sqrt(sum(b * b for b in job_vector))

    if mag1 == 0 or mag2 == 0:
        return 0.0

    return dot_product / (mag1 * mag2)