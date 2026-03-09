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
    page_title="JobFit",
    page_icon="◈",
    layout="wide",
)

# ── Design system ─────────────────────────────────────────────────────────────

st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap" rel="stylesheet">

<style>

/* ── Reset & base ── */
*, *::before, *::after { box-sizing: border-box; }

html, body, .stApp {
    font-family: 'DM Sans', sans-serif;
    background: #F9F8F6;
    color: #111;
}

/* Hide Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stAppViewContainer"] > .main {
    padding-left: 1rem !important;
    padding-right: 1rem !important;
}
.block-container {
    max-width: 820px !important;
    padding: 3rem 2rem 5rem !important;
    margin: 0 auto !important;
}

/* ── Custom header ── */
.jf-header {
    display: flex;
    justify-content: flex-end;
    align-items: center;
    gap: 10px;
    margin-bottom: 0.25rem;
}
.jf-header {
    position: absolute;
    top: 25px;
    right: 35px;
    display: flex;
    align-items: center;
    gap: 10px;
}
.jf-wordmark {
    font-size: 2rem;
    font-weight: 600;
    letter-spacing: -0.01em;
    color: #111;
}
.jf-badge {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    font-weight: 500;
    color: #888;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    padding: 2px 7px;
    border: 1px solid #ddd;
    border-radius: 2px;
}

/* ── Hero text ── */
.jf-hero {
    margin: 2.5rem 0 2rem;
}
.jf-hero h1 {
    font-size: 2.4rem;
    font-weight: 300;
    letter-spacing: -0.03em;
    line-height: 1.15;
    color: #111;
    margin: 0 0 0.5rem;
}
.jf-hero h1 em {
    font-style: normal;
    color: #C84B31;
}
.jf-hero p {
    font-size: 0.95rem;
    color: #666;
    font-weight: 400;
    margin: 0;
    line-height: 1.6;
}

/* ── Ruled divider ── */
.jf-rule {
    border: none;
    border-top: 1px solid #E0DDD8;
    margin: 2rem 0;
}

/* ── Form labels ── */
.jf-label {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #888;
    margin-bottom: 0.3rem;
}

/* ── Override Streamlit inputs ── */
.stTextInput > div > div > input,
.stSelectbox > div > div {
    background: #fff !important;
    border: 1px solid #E0DDD8 !important;
    border-radius: 4px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.9rem !important;
    color: #111 !important;
    box-shadow: none !important;
}
.stTextInput > div > div > input:focus {
    border-color: #C84B31 !important;
    outline: none !important;
}

/* ── File uploader ── */
.stFileUploader {
    background: #fff;
    border: 1px dashed #D0CCC6;
    border-radius: 6px;
    padding: 1.2rem;
    transition: border-color 0.2s;
}
.stFileUploader:hover {
    border-color: #C84B31;
}
[data-testid="stFileUploaderDropzoneInstructions"] {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.85rem !important;
    color: #888 !important;
}

/* ── Section heading ── */
.jf-section {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #888;
    margin: 2rem 0 1rem;
    display: flex;
    align-items: center;
    gap: 10px;
}
.jf-section::after {
    content: '';
    flex: 1;
    height: 1px;
    background: #E0DDD8;
}

/* ── Skill pills ── */
.jf-skills {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    margin-bottom: 0.5rem;
}
.jf-pill {
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    font-weight: 500;
    color: #444;
    background: #EFEDE9;
    border-radius: 3px;
    padding: 3px 9px;
    letter-spacing: 0.02em;
}

/* ── Job cards ── */
.jf-card {
    background: #fff;
    border: 1px solid #E8E5E0;
    border-left: 3px solid #E8E5E0;
    border-radius: 4px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 10px;
    transition: border-left-color 0.15s, box-shadow 0.15s;
    position: relative;
}
.jf-card:hover {
    border-left-color: #C84B31;
    box-shadow: 0 2px 16px rgba(0,0,0,0.05);
}
.jf-card.fit-good  { border-left-color: #2D6A4F; }
.jf-card.fit-maybe { border-left-color: #E6A817; }
.jf-card.fit-not   { border-left-color: #C0392B; }
.jf-card.knn-top   { border-left-color: #C84B31; }

.jf-card-top {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    margin-bottom: 0.4rem;
}
.jf-card-title {
    font-size: 1rem;
    font-weight: 600;
    color: #111;
    letter-spacing: -0.01em;
    margin: 0;
    line-height: 1.3;
}
.jf-card-company {
    font-size: 0.82rem;
    color: #666;
    margin: 2px 0 0;
    font-weight: 400;
}
.jf-card-location {
    font-size: 0.75rem;
    color: #999;
    margin-top: 1px;
}
.jf-score {
    font-family: 'DM Mono', monospace;
    font-size: 1.3rem;
    font-weight: 500;
    color: #111;
    white-space: nowrap;
    line-height: 1;
    text-align: right;
}
.jf-score-label {
    font-size: 0.62rem;
    color: #aaa;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    text-align: right;
    margin-top: 2px;
}

/* ── Progress bar ── */
.jf-bar-wrap {
    background: #F0EDE9;
    border-radius: 2px;
    height: 3px;
    margin: 10px 0 10px;
}
.jf-bar {
    height: 3px;
    border-radius: 2px;
    background: #111;
    transition: width 0.3s ease;
}
.jf-card.fit-good  .jf-bar { background: #2D6A4F; }
.jf-card.fit-maybe .jf-bar { background: #E6A817; }
.jf-card.fit-not   .jf-bar { background: #C0392B; }

/* ── Fit label ── */
.jf-fit {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    padding: 3px 9px;
    border-radius: 2px;
    margin-bottom: 8px;
}
.jf-fit.good  { background: #E8F5EE; color: #2D6A4F; }
.jf-fit.maybe { background: #FEF7E6; color: #A07800; }
.jf-fit.not   { background: #FDECEA; color: #C0392B; }

/* ── Matched skills in card ── */
.jf-matched {
    font-size: 0.78rem;
    color: #666;
    margin-top: 6px;
}
.jf-matched strong {
    color: #333;
    font-weight: 500;
}

/* ── KNN star badge ── */
.jf-knn {
    font-family: 'DM Mono', monospace;
    font-size: 0.62rem;
    font-weight: 500;
    background: #FEF0EC;
    color: #C84B31;
    border-radius: 2px;
    padding: 2px 6px;
    vertical-align: middle;
    margin-left: 6px;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}

/* ── Apply button ── */
.stLinkButton a {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    color: #111 !important;
    background: transparent !important;
    border: 1px solid #D0CCC6 !important;
    border-radius: 3px !important;
    padding: 5px 14px !important;
    letter-spacing: 0.04em !important;
    text-transform: uppercase !important;
    text-decoration: none !important;
    transition: background 0.15s, border-color 0.15s !important;
}
.stLinkButton a:hover {
    background: #111 !important;
    color: #fff !important;
    border-color: #111 !important;
}

/* ── Pagination ── */
.jf-page-info {
    font-family: 'DM Mono', monospace;
    font-size: 0.78rem;
    color: #999;
    text-align: center;
    padding-top: 8px;
    letter-spacing: 0.04em;
}
.stButton > button {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.8rem !important;
    font-weight: 500 !important;
    background: transparent !important;
    border: 1px solid #D0CCC6 !important;
    border-radius: 3px !important;
    color: #555 !important;
    padding: 6px 18px !important;
    transition: all 0.15s !important;
}
.stButton > button:hover:not(:disabled) {
    background: #111 !important;
    border-color: #111 !important;
    color: #fff !important;
}
.stButton > button:disabled {
    opacity: 0.3 !important;
}

/* ── KNN summary block ── */
.jf-knn-row {
    display: flex;
    align-items: baseline;
    gap: 12px;
    padding: 10px 0;
    border-bottom: 1px solid #EDEBE7;
}
.jf-knn-rank {
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    color: #bbb;
    min-width: 20px;
}
.jf-knn-title {
    font-size: 0.88rem;
    font-weight: 500;
    color: #111;
    flex: 1;
}
.jf-knn-company {
    font-size: 0.8rem;
    color: #888;
}
.jf-knn-dist {
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    color: #C84B31;
}

/* ── Warning / info overrides ── */
.stAlert {
    border-radius: 4px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.85rem !important;
}

/* ── Expander ── */
.stExpander {
    border: 1px solid #E8E5E0 !important;
    border-radius: 4px !important;
    background: #fff !important;
}

/* ── Caption / small text ── */
.jf-caption {
    font-size: 0.78rem;
    color: #999;
    margin-top: 0.25rem;
}

</style>
""", unsafe_allow_html=True)


# ── Naive Bayes ───────────────────────────────────────────────────────────────

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


# ── Header ────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="jf-header">
    <span class="jf-wordmark">◈ JobFit</span>
    <span class="jf-badge">AI · Canada</span>
</div>

<div class="jf-hero">
    <h1>Find roles that<br><em>actually</em> fit you.</h1>
    <p>Upload your resume. We rank live job postings by how well your skills match,<br>
    using cosine similarity and Naive Bayes classification.</p>
</div>

<hr class="jf-rule">
""", unsafe_allow_html=True)


# ── Search inputs ─────────────────────────────────────────────────────────────

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

col1, col2 = st.columns([3, 2])
with col1:
    st.markdown('<div class="jf-label">Role / Keywords</div>', unsafe_allow_html=True)
    search_term = st.text_input("Role", placeholder="e.g. Data Analyst, React Developer", label_visibility="collapsed")
with col2:
    st.markdown('<div class="jf-label">Location</div>', unsafe_allow_html=True)
    selected_city = st.selectbox("Location", canadian_cities, label_visibility="collapsed")

if not search_term.strip():
    st.markdown('<p class="jf-caption">Enter a job title or keywords above to get started.</p>', unsafe_allow_html=True)
    st.stop()

final_location = selected_city if selected_city else "Canada"
st.markdown(f'<p class="jf-caption">Searching in <strong>{final_location}</strong></p>', unsafe_allow_html=True)

# ── File uploader ─────────────────────────────────────────────────────────────

st.markdown('<div class="jf-label" style="margin-top:1.5rem">Resume</div>', unsafe_allow_html=True)
uploaded_file = st.file_uploader("Resume", type=["pdf", "docx"], label_visibility="collapsed")

# ── Pagination reset ──────────────────────────────────────────────────────────

current_query_key = f"{search_term.strip()}|{final_location}|{uploaded_file.name if uploaded_file else ''}"

if "last_query_key" not in st.session_state:
    st.session_state.last_query_key = current_query_key
if "page_number" not in st.session_state:
    st.session_state.page_number = 1
if st.session_state.last_query_key != current_query_key:
    st.session_state.page_number = 1
    st.session_state.last_query_key = current_query_key


# ── Main flow ─────────────────────────────────────────────────────────────────

if uploaded_file:

    # Step 1: Extract
    with st.spinner("Reading resume…"):
        resume_text = extract_text(uploaded_file)

    if not resume_text.strip():
        st.error("Could not extract text. Try a different PDF or DOCX.")
        st.stop()

    with st.expander("Extracted resume text", expanded=False):
        st.write(resume_text[:2000] + ("…" if len(resume_text) > 2000 else ""))

    # Step 2: Skills
    skills = extract_skills(resume_text)

    st.markdown('<div class="jf-section">Detected skills</div>', unsafe_allow_html=True)

    if skills:
        pills_html = '<div class="jf-skills">' + "".join(
            f'<span class="jf-pill">{s}</span>' for s in skills
        ) + "</div>"
        st.markdown(pills_html, unsafe_allow_html=True)
    else:
        st.warning("No recognised skills found. Check your resume mentions skills from our list.")

    st.markdown('<hr class="jf-rule">', unsafe_allow_html=True)

    # Step 3: Fetch & score
    with st.spinner("Fetching and ranking jobs…"):
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
        results.append({**job, "score": score, "fit_label": fit_label, "matched_skills": matched})

    results = [r for r in results if r["score"] > 0]
    ranked = sorted(results, key=lambda x: x["score"], reverse=True)

    # Step 4: KNN
    knn_results = knn_match(skills, results, k=3)
    knn_ids = [j["id"] for j in knn_results]

    # Step 5: Display
    if not ranked:
        st.warning("No matching jobs found. Try a different search term or location.")
        st.stop()

    st.markdown(
        f'<div class="jf-section">Matches <span style="font-weight:400;text-transform:none;'
        f'letter-spacing:0;color:#bbb;font-size:0.75rem;margin-left:4px">— {len(ranked)} results</span></div>',
        unsafe_allow_html=True
    )

    # Pagination
    page_size = 10
    total_pages = max(1, (len(ranked) + page_size - 1) // page_size)
    st.session_state.page_number = min(st.session_state.page_number, total_pages)
    start_idx = (st.session_state.page_number - 1) * page_size
    paged_results = ranked[start_idx:start_idx + page_size]

    # Job cards
    for result in paged_results:
        pct = round(result["score"] * 100)
        is_knn = result["id"] in knn_ids

        fit = result["fit_label"]
        card_class = "fit-good" if fit == "GoodFit" else ("fit-maybe" if fit == "Maybe" else "fit-not")
        if is_knn:
            card_class += " knn-top"

        fit_label_html = {
            "GoodFit": '<span class="jf-fit good">Good fit</span>',
            "Maybe":   '<span class="jf-fit maybe">Maybe</span>',
            "NotFit":  '<span class="jf-fit not">Not a fit</span>',
        }.get(fit, "")

        knn_badge = '<span class="jf-knn">KNN ★</span>' if is_knn else ""
        matched_text = ", ".join(result["matched_skills"]) if result["matched_skills"] else "none"
        bar_width = min(pct, 100)

        st.markdown(f"""
        <div class="jf-card {card_class}">
            <div class="jf-card-top">
                <div>
                    <div class="jf-card-title">{result['title']}{knn_badge}</div>
                    <div class="jf-card-company">{result['company']}</div>
                    <div class="jf-card-location">📍 {result.get('location', 'Unknown')}</div>
                </div>
                <div>
                    <div class="jf-score">{pct}%</div>
                    <div class="jf-score-label">match</div>
                </div>
            </div>
            <div class="jf-bar-wrap"><div class="jf-bar" style="width:{bar_width}%"></div></div>
            {fit_label_html}
            <div class="jf-matched"><strong>Skills matched:</strong> {matched_text}</div>
        </div>
        """, unsafe_allow_html=True)

        if result.get("url"):
            st.link_button("Apply →", result["url"])

    # Pagination controls
    st.markdown('<hr class="jf-rule" style="margin-top:1.5rem">', unsafe_allow_html=True)
    nav1, nav2, nav3 = st.columns([1, 2, 1])

    with nav1:
        if st.button("← Prev", disabled=st.session_state.page_number == 1):
            st.session_state.page_number -= 1
            st.rerun()

    with nav2:
        st.markdown(
            f'<div class="jf-page-info">{st.session_state.page_number} / {total_pages}</div>',
            unsafe_allow_html=True,
        )

    with nav3:
        if st.button("Next →", disabled=st.session_state.page_number == total_pages):
            st.session_state.page_number += 1
            st.rerun()

    # KNN section
    st.markdown('<div class="jf-section" style="margin-top:2.5rem">KNN top 3</div>', unsafe_allow_html=True)
    st.markdown(
        '<p class="jf-caption" style="margin-bottom:0.75rem">Closest matches by Euclidean distance in skill space.</p>',
        unsafe_allow_html=True
    )

    for i, job in enumerate(knn_results, 1):
        st.markdown(f"""
        <div class="jf-knn-row">
            <span class="jf-knn-rank">0{i}</span>
            <span class="jf-knn-title">{job['title']}</span>
            <span class="jf-knn-company">{job['company']}</span>
            <span class="jf-knn-dist">d={job['knn_distance']}</span>
        </div>
        """, unsafe_allow_html=True)