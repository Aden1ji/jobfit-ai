def build_job_vector(job_skills: list[str], vocab: list[str]) -> list[int]:
    """Convert a job's skill list into a binary vector over the shared vocab."""
    return [1 if skill in job_skills else 0 for skill in vocab]


def build_resume_vector(resume_skills: list[str], vocab: list[str]) -> list[int]:
    """Convert a resume's skill list into a binary vector over the shared vocab."""
    return [1 if skill in resume_skills else 0 for skill in vocab]


def euclidean_distance(vec_a: list[int], vec_b: list[int]) -> float:
    """Compute Euclidean distance between two vectors."""
    return sum((a - b) ** 2 for a, b in zip(vec_a, vec_b)) ** 0.5


def knn_match(resume_skills: list[str], jobs: list[dict], k: int = 3) -> list[dict]:
    """
    Ranks jobs by proximity to the resume using KNN (Euclidean distance).
    Smaller distance = better match.

    Args:
        resume_skills: list of skills extracted from the resume
        jobs: list of job dicts, each with a 'skills' key
        k: number of top matches to return

    Returns:
        Top k jobs sorted by closest distance (best match first)
    """
    # Build shared vocabulary from resume + all jobs
    all_skills = set(resume_skills)
    for job in jobs:
        all_skills.update(job["skills"])
    vocab = sorted(all_skills)

    resume_vector = build_resume_vector(resume_skills, vocab)

    ranked = []
    for job in jobs:
        job_vector = build_job_vector(job["skills"], vocab)
        dist = euclidean_distance(resume_vector, job_vector)
        ranked.append({**job, "knn_distance": round(dist, 4)})

    # Sort ascending — closer distance means better match
    ranked.sort(key=lambda x: x["knn_distance"])
    return ranked[:k]