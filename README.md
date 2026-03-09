# JobFit AI 

JobFit AI is a resume-powered job recommendation tool that helps users discover roles that match their skills.

The application analyzes a user's resume, extracts relevant skills, and ranks job postings based on how well those skills match job requirements.

The system combines multiple machine learning techniques to evaluate job fit and generate ranked recommendations.

---

# Overview

Finding relevant job opportunities can be difficult because many listings contain long descriptions and unclear skill requirements.

JobFit AI helps solve this problem by:

- Extracting skills directly from a resume
- Fetching real job postings from an external API
- Comparing resume skills with job requirements
- Ranking jobs based on similarity and machine learning models

This allows users to quickly identify positions that best match their experience.

---

# Features

## Resume Parsing
Extracts text from uploaded PDF or DOCX resumes.

## Skill Extraction
Identifies technical and professional skills from resume text.

## Job Search
Fetches live job postings using the Adzuna Jobs API.

## Similarity Scoring
Uses cosine similarity to compare resume skills with job requirements.

## Naive Bayes Classification
Predicts job fit categories such as:

- Good Fit
- Maybe
- Not a Fit

## K Nearest Neighbors Matching
Identifies the closest job matches based on skill space distance.

## Ranked Recommendations
Jobs are sorted from highest to lowest match score.

## Interactive Web Interface
Built with Streamlit to provide a simple and interactive user experience.

---

# Tech Stack

**Programming Language**

- Python

**Frontend**

- Streamlit

**Machine Learning**

- Cosine Similarity
- Naive Bayes
- K Nearest Neighbors

**API**

- Adzuna Job Search API

**Libraries**

- scikit-learn
- pandas
- numpy
- python-dotenv
- PDF / DOCX parsing libraries

---

# Project Structure

```
jobfit-ai
│
├── app.py
│
├── ai
│   ├── similarity.py
│   ├── naive_bayes.py
│   └── knn_matcher.py
│
├── services
│   ├── job_fetcher.py
│   ├── resume_parser.py
│   └── skill_extractor.py
│
├── utils
│   └── text_cleaner.py
│
├── data
│   └── skills_list.txt
│
└── README.md
```

---

# How It Works

**Step 1**  
The user uploads a resume file.

**Step 2**  
The resume text is extracted using the resume parser.

**Step 3**  
The skill extractor identifies known skills using a predefined skill list.

**Step 4**  
The job fetcher retrieves job postings using the Adzuna API.

**Step 5**  
Cosine similarity compares resume skills with job skills.

**Step 6**  
Naive Bayes predicts job fit labels.

**Step 7**  
KNN finds the closest job matches.

**Step 8**  
Results are ranked and displayed in the interface.

---

# Installation

## Clone the Repository

```bash
git clone https://github.com/Aden1ji/jobfit-ai.git
cd jobfit-ai
```

## Create Virtual Environment

```bash
python -m venv venv
```

### Activate Environment

Mac / Linux

```bash
source venv/bin/activate
```

Windows

```bash
venv\Scripts\activate
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Environment Variables

Create a `.env` file and add your Adzuna API credentials.

```
ADZUNA_APP_ID=your_app_id
ADZUNA_APP_KEY=your_api_key
```

You can obtain credentials from:

https://developer.adzuna.com/

---

# Running the Application

Start the Streamlit server:

```bash
streamlit run app.py
```

The app will open automatically in your browser.

---

# Example Workflow

1. Upload a resume
2. Enter a job keyword such as **Software Engineer** or **Data Analyst**
3. Choose a location
4. View ranked job recommendations
5. Apply directly through job links

---

# License

This is Group 10 CRN 74281 submission for Intro to AI Final Project
