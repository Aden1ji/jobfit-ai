import os
import requests
from dotenv import load_dotenv
from services.skill_extractor import extract_skills

load_dotenv()

APP_ID = os.getenv("ADZUNA_APP_ID")
APP_KEY = os.getenv("ADZUNA_APP_KEY")
COUNTRY = os.getenv("ADZUNA_COUNTRY", "ca")

# ── Fallback jobs (used when API is unavailable) ───────────────────────────────

FALLBACK_JOBS = [
    {
        "id": "J1",
        "title": "Software Developer",
        "company": "Tech Corp",
        "description": "python sql git rest api backend development",
        "skills": ["python", "sql", "git", "rest api"],
    },
    {
        "id": "J2",
        "title": "QA Analyst",
        "company": "QualitySoft",
        "description": "testing jira selenium git automation",
        "skills": ["testing", "jira", "selenium", "git"],
    },
    {
        "id": "J3",
        "title": "Data Analyst",
        "company": "Data Inc",
        "description": "sql excel power bi data analysis reporting",
        "skills": ["sql", "excel", "power bi", "data analysis"],
    },
    {
        "id": "J4",
        "title": "Frontend Developer",
        "company": "Web Labs",
        "description": "react javascript html css git frontend",
        "skills": ["react", "javascript", "html", "css", "git"],
    },
    {
        "id": "J5",
        "title": "Mobile Developer",
        "company": "AppWorks",
        "description": "react native firebase git javascript mobile",
        "skills": ["react native", "firebase", "git", "javascript"],
    },
]


def fetch_jobs(
    search: str = "software",
    location: str = "Canada",
    results_per_page: int = 20,
    max_pages: int = 5,
) -> list[dict]:
    """
    Fetch real job postings from Adzuna across multiple pages.
    Skills are extracted from each job description using your skills_list.txt.
    Falls back to local mock jobs if the API is unavailable or misconfigured.

    Returns jobs with keys: id, title, company, location, description, skills, url
    """
    if not APP_ID or not APP_KEY:
        print("[job_fetcher] Missing Adzuna credentials. Using fallback jobs.")
        return FALLBACK_JOBS

    all_jobs = []
    seen_ids = set()

    for page in range(1, max_pages + 1):
        url = f"https://api.adzuna.com/v1/api/jobs/{COUNTRY}/search/{page}"
        params = {
            "app_id": APP_ID,
            "app_key": APP_KEY,
            "results_per_page": results_per_page,
            "what": search,
            "where": location,
        }

        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            print(f"[job_fetcher] Page {page} failed: {e}. Stopping early.")
            break

        raw_jobs = data.get("results", [])
        if not raw_jobs:
            break  # No more results — stop fetching

        for job in raw_jobs:
            job_id = str(job.get("id", ""))
            if not job_id or job_id in seen_ids:
                continue  # Skip duplicates

            description = job.get("description", "")
            job_skills = extract_skills(description)

            if not job_skills:
                continue  # Skip jobs with no detectable skills

            all_jobs.append({
                "id": job_id,
                "title": job.get("title", "Unknown Title"),
                "company": job.get("company", {}).get("display_name", "Unknown Company"),
                "location": job.get("location", {}).get("display_name", ""),
                "description": description,
                "skills": job_skills,
                "url": job.get("redirect_url", ""),
            })
            seen_ids.add(job_id)

    if not all_jobs:
        print("[job_fetcher] No usable Adzuna jobs found. Using fallback jobs.")
        return FALLBACK_JOBS

    return all_jobs