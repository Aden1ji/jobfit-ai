import streamlit as st
import csv
from pathlib import Path

from services.resume_parser import extract_text
from services.skill_extractor import extract_skills
from services.job_fetcher import fetch_jobs
from ai.similarity import cosine_similarity
from ai.knn_matcher import knn_match
from ai.naive_bayes import NaiveBayes, make_features

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="JobFit AI",
    page_icon="⚡",
    layout="centered",
)

# ── Styling ───────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    .main { max-width: 780px; }
    .fit-good  { background:#d4f7e8; color:#1a7a4a; padding:3px 10px; border-radius:12px; font-weight:600; font-size:13px; }
    .fit-maybe { background:#fff3cd; color:#856404; padding:3px 10px; border-radius:12px; font-weight:600; font-size:13px; }
    .fit-not   { background:#fde8e8; color:#c0392b; padding:3px 10px; border-radius:12px; font-weight:600; font-size:13px; }
    .skill-tag { background:#e8f0fe; color:#1a56b0; padding:3px 10px; border-radius:8px;
                 font-size:12px; font-weight:600; display:inline-block; margin:2px; }
    .job-card  { border:1px solid #e0e0e0; border-radius:12px; padding:20px 24px;
                 margin-bottom:14px; background:#fafafa; }
    .score-bar-wrap { background:#eee; border-radius:6px; height:8px; margin-top:6px; }
    .score-bar { background:linear-gradient(90deg,#00b894,#00cec9); border-radius:6px; height:8px; }
</style>
""", unsafe_allow_html=True)


# ── Naive Bayes — train on labels.csv if it exists ───────────────────────────

@st.cache_resource
def load_model():
    labels_path = Path("data/labels.csv")
    if not labels_path.exists():
        return None

    jobs = {j["id"]: j for j in fetch_jobs()}

    training_resume_skills = {
        "R1": ["git", "javascript", "python", "react", "react native", "sql"],
        "R2": ["git", "jira", "selenium", "testing"],
        "R3": ["excel", "power bi", "sql"],
    }

    X, y = [], []
    with open(labels_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            r_skills = training_resume_skills.get(row["resume_id"], [])
            j_skills = jobs.get(row["job_id"], {}).get("skills", [])
            X.append(make_features(r_skills, j_skills))
            y.append(row["label"])

    model = NaiveBayes()
    model.fit(X, y)
    return model


model = load_model()


# ── UI ────────────────────────────────────────────────────────────────────────

st.markdown("## ⚡ JobFit AI")
st.markdown("Upload your resume and get ranked job recommendations powered by **Cosine Similarity** and **Naive Bayes**.")
st.divider()

canadian_cities = [
    "",
    "Toronto, ON", "Ottawa, ON", "Mississauga, ON", "Brampton, ON",
    "Hamilton, ON", "Oshawa, ON", "London, ON", "Waterloo, ON", "Windsor, ON",
    "Montreal, QC", "Quebec City, QC", "Laval, QC", "Gatineau, QC", "Sherbrooke, QC",
    "Calgary, AB", "Edmonton, AB", "Red Deer, AB",
    "Vancouver, BC", "Surrey, BC", "Burnaby, BC", "Richmond, BC", "Victoria, BC", "Kelowna, BC",
    "Saskatoon, SK", "Regina, SK",
    "Winnipeg, MB",
    "Halifax, NS",
    "St. John's, NL",
    "Fredericton, NB", "Moncton, NB",
    "Charlottetown, PE",
    "Whitehorse, YT",
    "Yellowknife, NT",
    "Iqaluit, NU",
]

col1, col2 = st.columns(2)
with col1:
    search_term = st.text_input("🔎 Job search")
with col2:
    selected_city = st.selectbox("📍 Pick a city", canadian_cities)

if not search_term.strip():
    st.warning("Please enter a job search term.")
    st.stop()

final_location = selected_city if selected_city else "Canada"

st.caption(f"Searching jobs in: **{final_location}**")

# ── File uploader comes BEFORE pagination reset so uploaded_file is defined ───
uploaded_file = st.file_uploader("📄 Upload your resume", type=["pdf", "docx"])

# ── Smart pagination reset — only resets when search actually changes ─────────
current_query_key = f"{search_term.strip()}|{final_location}|{uploaded_file.name if uploaded_file else ''}"

if "last_query_key" not in st.session_state:
    st.session_state.last_query_key = current_query_key

if "page_number" not in st.session_state:
    st.session_state.page_number = 1

if st.session_state.last_query_key != current_query_key:
    st.session_state.page_number = 1
    st.session_state.last_query_key = current_query_key

if uploaded_file:

    # ── Step 1: Extract text ─────────────────────────────────────────────────
    with st.spinner("Reading resume..."):
        resume_text = extract_text(uploaded_file)

    if not resume_text.strip():
        st.error("Could not extract text from this file. Try a different PDF or DOCX.")
        st.stop()

    with st.expander("📝 Extracted Resume Text", expanded=False):
        st.write(resume_text[:2000] + ("..." if len(resume_text) > 2000 else ""))

    # ── Step 2: Extract skills ───────────────────────────────────────────────
    skills = extract_skills(resume_text)

    st.markdown("### 🔍 Detected Skills")
    if skills:
        st.markdown(" ".join(f"`{s}`" for s in skills))
    else:
        st.warning("No recognised skills found. Check that your resume mentions skills from the skills list.")

    st.divider()

    # ── Step 3: Fetch jobs & score ───────────────────────────────────────────
    with st.spinner("Fetching jobs and ranking matches..."):
        jobs = fetch_jobs(
            search=search_term,
            location=final_location,
            results_per_page=20,
            max_pages=5,
        )

    results = []
    for job in jobs:
        score = cosine_similarity(skills, job["skills"])

        if model:
            features = make_features(skills, job["skills"])
            fit_label, _ = model.predict(features)
        else:
            overlap = len(set(skills) & set(job["skills"]))
            job_total = len(job["skills"])
            ratio = overlap / job_total if job_total else 0
            if ratio >= 0.7:
                fit_label = "GoodFit"
            elif ratio >= 0.4:
                fit_label = "Maybe"
            else:
                fit_label = "NotFit"

        matched = sorted(set(skills) & set(job["skills"]))

        results.append({
            **job,
            "score": score,
            "fit_label": fit_label,
            "matched_skills": matched,
        })

    # Filter out jobs with zero skill overlap — nothing to compare
    results = [r for r in results if r["score"] > 0]
    ranked = sorted(results, key=lambda x: x["score"], reverse=True)

    # ── Step 4: KNN top matches ──────────────────────────────────────────────
    knn_results = knn_match(skills, results, k=3)
    knn_ids = [j["id"] for j in knn_results]

    # ── Step 5: Display results ──────────────────────────────────────────────
    st.markdown("### 🎯 Ranked Job Matches")

    if not ranked:
        st.warning("No matching jobs found. Try a different search term or location.")
        st.stop()

    st.caption(f"Analyzed {len(ranked)} job postings · Sorted highest to lowest match · ⭐ = KNN top match")

    # ── Pagination ───────────────────────────────────────────────────────────
    page_size = 10

    total_pages = max(1, (len(ranked) + page_size - 1) // page_size)
    st.session_state.page_number = min(st.session_state.page_number, total_pages)

    start_idx = (st.session_state.page_number - 1) * page_size
    paged_results = ranked[start_idx:start_idx + page_size]

    # ── Job cards ────────────────────────────────────────────────────────────
    for result in paged_results:
        pct = round(result["score"] * 100)
        knn_star = " ⭐" if result["id"] in knn_ids else ""

        with st.container(border=True):
            top_left, top_right = st.columns([5, 2])

            with top_left:
                st.markdown(f"### {result['title']}{knn_star}")
                st.write(f"**{result['company']}**")
                st.caption(f"📍 {result.get('location', 'Unknown Location')}")

            with top_right:
                st.metric("Match Rating", f"{pct}%")

            st.progress(min(max(result["score"], 0.0), 1.0))

            if result["fit_label"] == "GoodFit":
                st.success("✅ Good Fit")
            elif result["fit_label"] == "Maybe":
                st.warning("⚠️ Maybe")
            else:
                st.error("❌ Not a Fit")

            matched_text = ", ".join(result["matched_skills"]) if result["matched_skills"] else "None"
            st.markdown(f"**Matched Skills:** {matched_text}")

            if result.get("url"):
                st.link_button("Apply Here", result["url"])

    # ── Pagination controls ──────────────────────────────────────────────────
    st.divider()
    nav1, nav2, nav3 = st.columns([1, 2, 1])

    with nav1:
        if st.button("◀ Previous", disabled=st.session_state.page_number == 1):
            st.session_state.page_number -= 1
            st.rerun()

    with nav2:
        st.markdown(
            f"<div style='text-align:center; padding-top:6px;'>Page {st.session_state.page_number} of {total_pages}</div>",
            unsafe_allow_html=True,
        )

    with nav3:
        if st.button("Next ▶", disabled=st.session_state.page_number == total_pages):
            st.session_state.page_number += 1
            st.rerun()

    # ── KNN section ──────────────────────────────────────────────────────────
    st.divider()
    st.markdown("### 📊 KNN Closest Matches (Top 3)")
    st.caption("Based on Euclidean distance in skill space — lower distance = better match")
    for i, job in enumerate(knn_results, 1):
        st.markdown(f"**{i}. {job['title']}** — {job['company']} &nbsp; `distance: {job['knn_distance']}`",
                    unsafe_allow_html=True)