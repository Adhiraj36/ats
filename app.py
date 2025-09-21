import streamlit as st
import pandas as pd
import numpy as np
import re
import io
import PyPDF2
import docx2txt
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer

# --- NLTK Data Download ---
# Ensures that the necessary NLTK models are available.
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt', quiet=True)
try:
    nltk.data.find('corpora/stopwords')
except nltk.downloader.DownloadError:
    nltk.download('stopwords', quiet=True)

# --- Core Application Functions ---

@st.cache_resource
def load_embedding_engine():
    """
    Loads and caches the Sentence Transformer model. Using a lightweight,
    efficient model for generating high-quality text embeddings.
    """
    return SentenceTransformer('all-MiniLM-L6-v2')

def extract_text_from_document(uploaded_file: io.BytesIO) -> str:
    """
    Extracts raw text from an uploaded file (PDF or DOCX).
    Handles file reading and text extraction based on file extension.
    """
    if uploaded_file is None:
        return ""

    name = uploaded_file.name.lower()
    text_content = ""

    try:
        if name.endswith(".pdf"):
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            for page in pdf_reader.pages:
                if page.extract_text():
                    text_content += page.extract_text() + "\n"
        elif name.endswith((".docx", ".doc")):
            text_content = docx2txt.process(uploaded_file)
    except Exception as e:
        st.error(f"Error reading {uploaded_file.name}: {e}")
        return ""
    return text_content.strip()

def clean_and_normalize_text(text: str) -> str:
    """
    Preprocesses raw text by converting to lowercase, removing special characters,
    and normalizing whitespace. Essential for accurate analysis.
    """
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def identify_technical_skills(text: str) -> List[str]:
    """
    Identifies a predefined list of technical skills from text using regex.
    This list can be expanded to include more domain-specific skills.
    """
    skill_patterns = [
        r'\b(python|java|javascript|c\+\+|c#|php|ruby|go|swift|kotlin|typescript)\b',
        r'\b(react|angular|vue|node\.?js|django|flask|spring|fastapi)\b',
        r'\b(mysql|postgresql|mongodb|redis|cassandra|sqlite)\b',
        r'\b(aws|azure|gcp|google cloud|docker|kubernetes|terraform|jenkins)\b',
        r'\b(machine learning|deep learning|data science|pandas|numpy|scikit-learn|tensorflow|pytorch)\b',
        r'\b(git|jira|agile|scrum|rest api|graphql|microservices)\b'
    ]
    identified_skills = set()
    for pattern in skill_patterns:
        identified_skills.update(re.findall(pattern, text.lower()))
    return sorted(list(identified_skills))

def compute_keyword_score(resume_text: str, jd_text: str, top_k: int = 50) -> Tuple[float, List[str]]:
    """
    Calculates a score based on TF-IDF keyword overlap between resume and JD.
    """
    try:
        clean_resume = clean_and_normalize_text(resume_text)
        clean_jd = clean_and_normalize_text(jd_text)

        stop_words = set(stopwords.words('english'))
        vectorizer = TfidfVectorizer(max_features=top_k, stop_words=list(stop_words))

        jd_tfidf = vectorizer.fit_transform([clean_jd])
        jd_keywords = set(vectorizer.get_feature_names_out())

        if not jd_keywords:
            return 0.0, []

        resume_tokens = word_tokenize(clean_resume)
        resume_keywords = set(w for w in resume_tokens if w in jd_keywords)

        overlap = len(resume_keywords)
        score = (overlap / len(jd_keywords)) * 100
        return min(score, 100.0), list(jd_keywords - resume_keywords)
    except Exception:
        return 0.0, []


def compute_semantic_score(resume_embedding: np.ndarray, jd_embedding: np.ndarray) -> float:
    """
    Computes semantic similarity score using cosine similarity between embeddings.
    Normalizes the score to be between 0 and 100.
    """
    if resume_embedding is None or jd_embedding is None:
        return 0.0
    similarity = cosine_similarity(resume_embedding.reshape(1, -1), jd_embedding.reshape(1, -1))[0][0]
    return (similarity + 1) / 2 * 100  # Normalize from [-1, 1] to [0, 100]

def generate_text_embedding(text: str, model) -> Optional[np.ndarray]:
    """
    Generates a sentence embedding for a given text using the loaded model.
    """
    if not text:
        return None
    try:
        return model.encode(text.replace("\n", " "), convert_to_numpy=True)
    except Exception as e:
        st.warning(f"Could not generate embedding: {e}")
        return None

def analyze_candidate_suitability(resume_content: str, jd_content: str, model, weights: Dict[str, float]):
    """
    Main evaluation function. Orchestrates text extraction, scoring,
    and generation of suggestions for a single resume.
    """
    # Keyword & Skill Analysis
    resume_skills = identify_technical_skills(resume_content)
    jd_skills = identify_technical_skills(jd_content)
    skill_overlap = set(resume_skills) & set(jd_skills)
    missing_skills = set(jd_skills) - set(resume_skills)
    skill_score = (len(skill_overlap) / len(jd_skills)) * 100 if jd_skills else 100.0
    keyword_score, missing_keywords = compute_keyword_score(resume_content, jd_content)

    hard_match_score = (skill_score * 0.6) + (keyword_score * 0.4)

    # Semantic Analysis
    resume_embedding = generate_text_embedding(resume_content, model)
    jd_embedding = generate_text_embedding(jd_content, model)
    semantic_score = compute_semantic_score(resume_embedding, jd_embedding)

    # Final Weighted Score
    final_score = (hard_match_score * weights['hard']) + (semantic_score * weights['semantic'])

    # Verdict
    if final_score >= 75:
        verdict = "High Suitability"
    elif final_score >= 50:
        verdict = "Medium Suitability"
    else:
        verdict = "Low Suitability"

    return {
        "final_score": min(final_score, 100.0),
        "hard_score": min(hard_match_score, 100.0),
        "semantic_score": semantic_score,
        "verdict": verdict,
        "analysis": {
            "resume_skills": sorted(list(resume_skills)),
            "jd_skills": sorted(list(jd_skills)),
            "missing_skills": sorted(list(missing_skills)),
            "missing_keywords": sorted(list(missing_keywords))[:5], # Show top 5 missing keywords
        },
        "timestamp": datetime.now().isoformat()
    }

# --- Streamlit UI Components ---

def render_ui_styling():
    """
    Applies custom CSS for ambiance, animations, and cross-browser compatibility.
    """
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&display=swap');

    html, body, [class*="st-"] {
        font-family: 'Roboto', sans-serif;
    }

    /* Animated Gradient Background */
    .stApp {
        background: linear-gradient(-45deg, #0d1b2a, #1b263b, #415a77, #778da9);
        background-size: 400% 400%;
        animation: gradientBG 15s ease infinite;
        color: #E0E1DD;
    }

    @keyframes gradientBG {
        0% {background-position: 0% 50%;}
        50% {background-position: 100% 50%;}
        100% {background-position: 0% 50%;}
    }

    /* Main content styling */
    .block-container {
        padding: 2rem 2rem 3rem 2rem;
    }

    /* Card-like styling for containers */
    .st-emotion-cache-1r4qj8v, .st-emotion-cache-z5fcl4 {
        background-color: rgba(255, 255, 255, 0.05);
        border-radius: 15px;
        padding: 1.5rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(4px);
        -webkit-backdrop-filter: blur(4px);
    }

    /* Button Styling */
    .stButton>button {
        background: linear-gradient(45deg, #778da9, #415a77);
        color: white;
        font-weight: bold;
        border-radius: 10px;
        padding: 12px 24px;
        border: none;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
        transition: all 0.3s ease-in-out;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.3);
        background: linear-gradient(45deg, #8a9cb4, #526c8e);
    }

    /* Headers and Titles */
    h1, h2, h3 {
        color: #FFFFFF;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }

    /* Suppress unwanted default tooltips that cause UI glitches */
    [data-testid="stFileUploadDropzone"] > div > div > span {
        visibility: hidden;
        position: relative;
    }
    [data-testid="stFileUploadDropzone"] > div > div > span::after {
        visibility: visible;
        content: 'Drag and drop file here or click to browse';
        display: block;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

def get_verdict_styling(verdict: str) -> str:
    """Returns CSS styling based on the suitability verdict."""
    colors = {
        "High Suitability": "background-color: #2a9d8f; color: white; font-weight: bold;",
        "Medium Suitability": "background-color: #e9c46a; color: #333; font-weight: bold;",
        "Low Suitability": "background-color: #e76f51; color: white; font-weight: bold;"
    }
    return colors.get(verdict, "")

def display_evaluation_results(results: List[Dict]):
    """
    Renders the results in a structured and visually appealing format.
    """
    st.subheader("📊 Evaluation Summary")
    summary_df = pd.DataFrame([{
        "Resume": r["filename"],
        "Final Score": f"{r['final_score']:.1f}",
        "Verdict": r["verdict"],
    } for r in results])

    st.dataframe(
        summary_df.style.apply(
            lambda row: [get_verdict_styling(row['Verdict'])] * len(row), axis=1
        ),
        use_container_width=True
    )

    for r in results:
        with st.expander(f"### Detailed Analysis for: {r['filename']}"):
            col1, col2, col3 = st.columns(3)
            col1.metric("Final Score", f"{r['final_score']:.1f}/100")
            col2.metric("Keyword & Skill Match", f"{r['hard_score']:.1f}/100")
            col3.metric("Semantic Context Match", f"{r['semantic_score']:.1f}/100")

            st.markdown(f"**Verdict:** <span style='{get_verdict_styling(r['verdict'])} padding: 5px 10px; border-radius: 5px;'>{r['verdict']}</span>", unsafe_allow_html=True)
            st.markdown("---")

            st.subheader("Improvement Suggestions")
            if r['analysis']['missing_skills']:
                st.warning(f"**Missing Skills:** Consider adding: {', '.join(r['analysis']['missing_skills'])}")
            if r['analysis']['missing_keywords']:
                st.warning(f"**Missing Keywords:** Your resume could be strengthened by including terms like: {', '.join(r['analysis']['missing_keywords'])}")
            if r['semantic_score'] < 60:
                st.info("**Content Alignment:** Rephrase your experience to better match the language and context of the job description.")
            st.success("**General Tip:** Always quantify your achievements with numbers and metrics where possible (e.g., 'Increased efficiency by 20%').")


def render_evaluation_page(model):
    """Renders the main page for uploading documents and evaluating resumes."""
    st.header("📄 Evaluate New Candidates")

    with st.container():
        st.subheader("1. Provide the Job Description")
        jd_content_text = st.text_area("Paste the Job Description here:", height=200, key="jd_text")
        # The 'help' parameter is what can cause the strange keyword tooltip.
        # The CSS above provides a fix.
        jd_file = st.file_uploader(
            "Or upload a Job Description file:",
            type=['pdf','docx'],
            help="Upload a single PDF or DOCX file for the job description."
        )

        jd_content = jd_content_text
        if jd_file:
            jd_content = extract_text_from_document(jd_file)
            if not jd_content:
                st.warning("Could not extract text from the uploaded JD file.")

    if not jd_content:
        st.warning("Please provide a job description to begin.")
        return

    with st.container():
        st.subheader("2. Upload Candidate Resumes")
        resume_files = st.file_uploader(
            "Upload one or more resume files:",
            type=['pdf', 'docx'],
            accept_multiple_files=True,
            help="You can upload multiple PDF or DOCX resume files at once."
        )

    with st.container():
        st.subheader("3. Configure Scoring Weights")
        col1, col2 = st.columns(2)
        hard_weight = col1.slider("Keyword & Skill Match Weight", 0.0, 1.0, 0.4, 0.05)
        semantic_weight = col2.slider("Semantic Context Match Weight", 0.0, 1.0, 0.6, 0.05)
        weights = {'hard': hard_weight, 'semantic': semantic_weight}

    if st.button("🚀 Analyze Resumes", use_container_width=True):
        if not resume_files:
            st.error("Please upload at least one resume.")
            return

        with st.spinner("Analyzing... This may take a moment."):
            results = []
            progress_bar = st.progress(0)
            for i, file in enumerate(resume_files):
                resume_text = extract_text_from_document(file)
                if resume_text:
                    analysis = analyze_candidate_suitability(resume_text, jd_content, model, weights)
                    analysis['filename'] = file.name
                    results.append(analysis)
                    st.session_state.evaluations.append(analysis)
                progress_bar.progress((i + 1) / len(resume_files))

        if results:
            display_evaluation_results(results)

def render_dashboard_page():
    """Renders the dashboard for visualizing and filtering past evaluations."""
    st.header("📈 Evaluation Dashboard")

    if not st.session_state.evaluations:
        st.info("No evaluations have been performed yet. Go to the 'Evaluate' page to start.")
        return

    evaluations = st.session_state.evaluations
    df = pd.DataFrame(evaluations)
    df['final_score'] = df['final_score'].astype(float) # Ensure correct type for filtering

    # --- Filtering Controls ---
    st.subheader("Filter Results")
    col1, col2 = st.columns(2)
    min_score = col1.slider("Minimum Final Score", 0, 100, 50)
    verdict_options = ["All"] + list(df['verdict'].unique())
    verdict_filter = col2.selectbox("Filter by Verdict", options=verdict_options)

    filtered_df = df[df['final_score'] >= min_score]
    if verdict_filter != "All":
        filtered_df = filtered_df[filtered_df['verdict'] == verdict_filter]

    if filtered_df.empty:
        st.info("No results match the current filters.")
        return

    # --- Visualizations ---
    st.subheader("Candidate Score Comparison")
    chart_df = filtered_df[['filename', 'final_score', 'hard_score', 'semantic_score']].rename(columns={'filename': 'Candidate'})
    st.bar_chart(chart_df.set_index('Candidate'))

    st.subheader("Filtered Data")
    display_df = filtered_df[['filename', 'final_score', 'verdict']].copy()
    display_df['final_score'] = display_df['final_score'].apply(lambda x: f"{x:.1f}")

    st.dataframe(
        display_df.style.apply(lambda row: [get_verdict_styling(row['verdict'])] * len(row), axis=1),
        use_container_width=True
    )

# --- Main Application Logic ---

def main():
    """
    Main function to run the Streamlit application.
    Sets up the page, loads models, and handles navigation.
    """
    st.set_page_config(
        page_title="AI Resume Relevance Checker",
        page_icon="🎯",
        layout="wide"
    )
    render_ui_styling()
    st.title("🎯 AI Resume Relevance Checker")
    st.markdown("---")

    # Initialize session state
    if "evaluations" not in st.session_state:
        st.session_state.evaluations = []

    # Load resources
    embedding_model = load_embedding_engine()

    # Sidebar Navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Evaluate Resumes", "Evaluation Dashboard"])
    st.sidebar.markdown("---")
    st.sidebar.info(
        "This tool uses AI to analyze resume suitability. "
        "Scores are indicative and should be used to support, not replace, "
        "human judgment in the hiring process."
    )

    if page == "Evaluate Resumes":
        render_evaluation_page(embedding_model)
    else:
        render_dashboard_page()

if __name__ == "__main__":
    main()

