import streamlit as st
import pandas as pd
import numpy as np
import re
import PyPDF2
import docx2txt
from datetime import datetime
from typing import List, Dict, Optional
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Using a local, open-source model for embeddings
from sentence_transformers import SentenceTransformer

# Download necessary NLTK data if it's not already present
# This is for tokenizing words and removing common stopwords.
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
except Exception as e:
    st.warning(f"Could not download NLTK data. Keyword extraction might be affected. Error: {e}")

# Cache the embedding model so it doesn't reload on every interaction.
@st.cache_resource
def load_embedding_model():
    """Loads the SentenceTransformer model from cache."""
    return SentenceTransformer('all-MiniLM-L6-v2')

def extract_text_from_pdf(uploaded_file) -> str:
    """Extracts text content from an uploaded PDF file."""
    try:
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        text = ""
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text.strip()
    except Exception as e:
        st.error(f"Error extracting text from PDF: {str(e)}")
        return ""

def extract_text_from_docx(uploaded_file) -> str:
    """Extracts text content from an uploaded DOCX file."""
    try:
        return docx2txt.process(uploaded_file).strip()
    except Exception as e:
        st.error(f"Error extracting text from DOCX: {str(e)}")
        return ""

def extract_text_from_file(uploaded_file) -> str:
    """Determines the file type and calls the appropriate text extraction function."""
    if uploaded_file is None:
        return ""
    file_name = uploaded_file.name.lower()
    if file_name.endswith(".pdf"):
        return extract_text_from_pdf(uploaded_file)
    elif file_name.endswith((".docx", ".doc")):
        return extract_text_from_docx(uploaded_file)
    else:
        st.error(f"Unsupported file type: {file_name}. Please upload PDF or DOCX files.")
        return ""

def preprocess_text(text: str) -> str:
    """Cleans text by lowercasing, removing special characters, and normalizing whitespace."""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text) # Keep only alphanumeric and spaces
    text = re.sub(r'\s+', ' ', text) # Replace multiple spaces with a single one
    return text.strip()

def extract_keywords(text: str, top_k: int = 20) -> List[str]:
    """Extracts the most important keywords from text using TF-IDF."""
    try:
        cleaned_text = preprocess_text(text)
        stop_words = set(stopwords.words('english'))
        
        # Tokenize and filter out stopwords
        words = word_tokenize(cleaned_text)
        filtered_words = [word for word in words if word not in stop_words and len(word) > 2]
        
        if not filtered_words:
            return []
            
        # Use TF-IDF to find the most relevant words
        vectorizer = TfidfVectorizer(max_features=top_k, stop_words='english')
        tfidf_matrix = vectorizer.fit_transform([' '.join(filtered_words)])
        feature_names = vectorizer.get_feature_names_out()
        return feature_names.tolist()
    except Exception as e:
        st.warning(f"Keyword extraction failed: {str(e)}")
        return []

def extract_skills(text: str) -> List[str]:
    """Identifies a predefined list of technical skills in the text."""
    # A more robust solution might use a larger skills database or a trained NER model
    skill_patterns = [
        r'\b(python|java|javascript|c\+\+|c#|php|ruby|go|swift|kotlin|scala|r)\b',
        r'\b(html|css|react|angular|vue|node\.?js|express|django|flask|spring)\b',
        r'\b(mysql|postgresql|mongodb|redis|elasticsearch|oracle|sql server)\b',
        r'\b(aws|azure|gcp|google cloud|kubernetes|docker|terraform)\b',
        r'\b(machine learning|deep learning|data science|pandas|numpy|tensorflow|pytorch)\b',
        r'\b(git|jenkins|agile|scrum|rest|api|microservices)\b'
    ]
    text_lower = text.lower()
    found_skills = set()
    for pattern in skill_patterns:
        found_skills.update(re.findall(pattern, text_lower, re.IGNORECASE))
    return sorted(list(found_skills))

# Cache the embedding generation for performance
@st.cache_data
def get_embedding(text: str, _model) -> Optional[np.ndarray]:
    """Generates a sentence embedding for the given text."""
    if not text.strip() or _model is None:
        return None
    try:
        # The model expects a single string
        embedding = _model.encode(text.replace("\n", " "), convert_to_numpy=True)
        return embedding
    except Exception as e:
        st.error(f"Embedding generation failed: {str(e)}")
        return None

def cosine_sim(v1: np.ndarray, v2: np.ndarray) -> float:
    """Calculates the cosine similarity between two vectors."""
    if v1 is None or v2 is None:
        return 0.0
    # Reshape for scikit-learn's cosine_similarity function
    return float(cosine_similarity(v1.reshape(1, -1), v2.reshape(1, -1))[0][0])

def calculate_hard_match(resume_text: str, jd_text: str) -> Dict:
    """Calculates scores based on keyword and skill overlap."""
    resume_keywords = set(extract_keywords(resume_text, 50))
    jd_keywords = set(extract_keywords(jd_text, 50))
    resume_skills = set(extract_skills(resume_text))
    jd_skills = set(extract_skills(jd_text))

    keyword_overlap = len(resume_keywords.intersection(jd_keywords))
    skill_overlap = len(resume_skills.intersection(jd_skills))

    # Calculate scores as a percentage of required items found
    keyword_score = (keyword_overlap / max(len(jd_keywords), 1)) * 100
    skill_score = (skill_overlap / max(len(jd_skills), 1)) * 100
    
    missing_skills = sorted(list(jd_skills - resume_skills))

    # Combine scores with a 60/40 weight for keywords/skills
    hard_score = (keyword_score * 0.6) + (skill_score * 0.4)
    
    return {
        "score": min(hard_score, 100),
        "keyword_overlap": keyword_overlap,
        "skills_overlap": skill_overlap,
        "missing_skills": missing_skills,
        "resume_skills": sorted(list(resume_skills)),
        "jd_skills": sorted(list(jd_skills)),
    }

def calculate_semantic_score(resume_emb: np.ndarray, jd_emb: np.ndarray) -> float:
    """Calculates a score based on the semantic similarity of embeddings."""
    similarity = cosine_sim(resume_emb, jd_emb)
    # Normalize similarity from [-1, 1] to a score of [0, 100]
    return ((similarity + 1) / 2) * 100

def get_final_score(hard_score: float, semantic_score: float, hard_w: float, semantic_w: float) -> float:
    """Combines hard and semantic scores using user-defined weights."""
    # Ensure weights sum to 1 for a proper weighted average
    total_weight = hard_w + semantic_w
    if total_weight == 0: return 0.0
    
    score = (hard_score * hard_w + semantic_score * semantic_w) / total_weight
    return min(score, 100)

def get_verdict(score: float) -> str:
    """Assigns a suitability verdict based on the final score."""
    if score >= 75:
        return "High Suitability"
    elif score >= 50:
        return "Medium Suitability"
    else:
        return "Low Suitability"

def generate_suggestions(hard_match_results: Dict, semantic_score: float) -> List[str]:
    """Provides actionable suggestions for improving the resume."""
    suggestions = []
    missing_skills = hard_match_results.get('missing_skills', [])
    
    if missing_skills:
        suggestions.append(f"Incorporate missing skills: {', '.join(missing_skills[:5])}.")
    if hard_match_results['score'] < 60:
        suggestions.append("Increase the overlap of keywords from the job description.")
    if semantic_score < 60:
        suggestions.append("Rephrase your experience to better match the language and tone of the job description.")
    if not hard_match_results.get('resume_skills'):
        suggestions.append("Clearly list your technical skills in a dedicated section.")
    
    suggestions.append("Quantify achievements with metrics (e.g., 'Increased efficiency by 20%').")
    return suggestions

def evaluate_resume(resume_text: str, jd_text: str, embed_model, hard_w: float, semantic_w: float) -> Dict:
    """Runs the full evaluation pipeline for a single resume."""
    resume_emb = get_embedding(resume_text, embed_model)
    jd_emb = get_embedding(jd_text, embed_model)

    hard_match_results = calculate_hard_match(resume_text, jd_text)
    semantic_score = calculate_semantic_score(resume_emb, jd_emb)
    
    final_score = get_final_score(hard_match_results['score'], semantic_score, hard_w, semantic_w)
    verdict = get_verdict(final_score)
    suggestions = generate_suggestions(hard_match_results, semantic_score)

    return {
        "final_score": final_score,
        "hard_score": hard_match_results['score'],
        "semantic_score": semantic_score,
        "verdict": verdict,
        "skills_analysis": hard_match_results,
        "suggestions": suggestions,
        "timestamp": datetime.now().isoformat()
    }

def get_verdict_color(verdict: str) -> str:
    """Returns a background color based on the verdict for UI styling."""
    return {
        "High Suitability": "#28a745", # Green
        "Medium Suitability": "#ffc107", # Yellow
        "Low Suitability": "#dc3545" # Red
    }.get(verdict, "#6c757d") # Gray for others

# --- Streamlit UI ---
def main():
    st.set_page_config(page_title="AI Resume Relevance Checker", page_icon="🎯", layout="wide")

    # Custom CSS for a more polished, ambient look
    st.markdown("""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&display=swap');
            
            html, body, [class*="st-"] {
                font-family: 'Roboto', sans-serif;
            }
            
            .stApp {
                background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            }
            
            .stButton>button {
                background-color: #4B8BBE;
                color: white;
                font-weight: bold;
                border-radius: 8px;
                border: none;
                transition: all 0.3s ease-in-out;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            }
            .stButton>button:hover {
                background-color: #306998;
                transform: translateY(-2px);
                box-shadow: 0 6px 8px rgba(0,0,0,0.15);
            }

            .stExpander {
                border: 1px solid #e6e6e6;
                border-radius: 10px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.05);
            }
        </style>
    """, unsafe_allow_html=True)
    
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Evaluate Resumes", "Evaluation Dashboard"])

    if "evaluations" not in st.session_state:
        st.session_state.evaluations = []
    
    embed_model = load_embedding_model()

    if page == "Evaluate Resumes":
        st.title("🎯 AI-Powered Resume Relevance Checker")
        st.markdown("Evaluate how well resumes align with a job description.")

        with st.container():
            st.header("1. Job Description")
            jd_text_area = st.text_area("Paste the job description here:", height=150, key="jd_text")
            jd_file = st.file_uploader("Or upload a Job Description file (PDF or DOCX)", type=['pdf', 'docx'])
            
            job_desc_text = ""
            if jd_file:
                job_desc_text = extract_text_from_file(jd_file)
            elif jd_text_area:
                job_desc_text = jd_text_area
        
        with st.container():
            st.header("2. Resumes")
            uploaded_resumes = st.file_uploader(
                "Upload one or more resume files (PDF or DOCX):",
                type=['pdf', 'docx'],
                accept_multiple_files=True
            )
        
        with st.container():
            st.header("3. Configure & Evaluate")
            col1, col2 = st.columns(2)
            with col1:
                hard_w = st.slider("Weight for Keyword/Skill Match:", 0.0, 1.0, 0.4, 0.05)
            with col2:
                semantic_w = st.slider("Weight for Semantic Match:", 0.0, 1.0, 0.6, 0.05)

            if st.button("🚀 Evaluate Resumes", use_container_width=True):
                if not job_desc_text.strip():
                    st.error("Please provide a job description before evaluating.")
                    return
                if not uploaded_resumes:
                    st.error("Please upload at least one resume to evaluate.")
                    return

                progress_bar = st.progress(0, text="Starting evaluation...")
                new_results = []
                for i, resume_file in enumerate(uploaded_resumes):
                    progress_bar.progress((i + 1) / len(uploaded_resumes), text=f"Processing {resume_file.name}...")
                    resume_text = extract_text_from_file(resume_file)
                    if not resume_text:
                        st.warning(f"Could not extract text from {resume_file.name}. Skipping.")
                        continue
                    
                    eval_result = evaluate_resume(resume_text, job_desc_text, embed_model, hard_w, semantic_w)
                    eval_result['filename'] = resume_file.name
                    new_results.append(eval_result)
                
                st.session_state.evaluations.extend(new_results)
                progress_bar.empty()
                st.success(f"Evaluation complete for {len(new_results)} resume(s)!")

        if st.session_state.evaluations:
            st.header("📊 Evaluation Results")
            for r in reversed(st.session_state.evaluations):
                with st.expander(f"**{r['filename']}** | Score: {r['final_score']:.1f} | Verdict: {r['verdict']}"):
                    sc1, sc2, sc3 = st.columns(3)
                    sc1.metric("Final Score", f"{r['final_score']:.1f}/100")
                    sc2.metric("Hard Match Score", f"{r['hard_score']:.1f}/100")
                    sc3.metric("Semantic Match Score", f"{r['semantic_score']:.1f}/100")

                    st.markdown(f"**Verdict:** <span style='color:white; background-color:{get_verdict_color(r['verdict'])}; padding: 3px 8px; border-radius: 5px;'>{r['verdict']}</span>", unsafe_allow_html=True)
                    st.subheader("Improvement Suggestions")
                    for sug in r['suggestions']:
                        st.write(f"- {sug}")

                    st.subheader("Skills Analysis")
                    sa1, sa2 = st.columns(2)
                    with sa1:
                        st.write("**Required Skills (from JD):**")
                        st.write(r['skills_analysis']['jd_skills'] or "_None identified_")
                    with sa2:
                        st.write("**Missing Skills:**")
                        st.write(r['skills_analysis']['missing_skills'] or "_None_")

    elif page == "Evaluation Dashboard":
        st.title("📈 Evaluation Dashboard")
        if not st.session_state.evaluations:
            st.info("No evaluations have been performed yet. Go to the 'Evaluate Resumes' page to start.")
            return

        all_evals = st.session_state.evaluations
        
        # --- Filters ---
        st.subheader("Filters")
        col1, col2 = st.columns(2)
        with col1:
            min_score = st.slider("Minimum Final Score", 0, 100, 0)
        with col2:
            verdicts = ["High Suitability", "Medium Suitability", "Low Suitability"]
            verdict_filter = st.multiselect("Filter by Verdict", options=verdicts, default=verdicts)

        filtered_evals = [e for e in all_evals if e['final_score'] >= min_score and e['verdict'] in verdict_filter]

        if not filtered_evals:
            st.warning("No results match the current filters.")
            return
            
        # --- Comparison Graph ---
        st.subheader("Score Comparison Chart")
        chart_data = pd.DataFrame([{
            "Resume": r["filename"],
            "Final Score": r["final_score"],
            "Hard Match": r["hard_score"],
            "Semantic Match": r["semantic_score"],
        } for r in filtered_evals]).set_index("Resume")
        st.bar_chart(chart_data)

        # --- Results Table ---
        st.subheader("Filtered Results")
        df_data = [{
            "Resume": r["filename"],
            "Score": f"{r['final_score']:.1f}",
            "Verdict": r["verdict"],
            "Hard Match": f"{r['hard_score']:.1f}",
            "Semantic Match": f"{r['semantic_score']:.1f}",
            "Missing Skills": len(r["skills_analysis"]["missing_skills"])
        } for r in filtered_evals]
        df = pd.DataFrame(df_data)

        def style_verdict(row):
            color = get_verdict_color(row['Verdict'])
            return [f'background-color: {color}; color: white' if col == 'Verdict' else '' for col in row.index]
        st.dataframe(df.style.apply(style_verdict, axis=1), use_container_width=True)

        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Filtered Results as CSV",
            data=csv,
            file_name=f"resume_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
        )

if __name__ == "__main__":
    main()
