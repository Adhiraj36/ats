import streamlit as st
import pandas as pd
import numpy as np
import re
import io
import PyPDF2
import docx2txt
from datetime import datetime
from typing import List, Dict, Optional
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from openai import OpenAI

# Download necessary NLTK data if not already
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
except:
    pass

# Initialize OpenAI client
@st.cache_resource
def init_openai_client():
    import os
    api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("OpenAI API key not found! Please set OPENAI_API_KEY in secrets or environment.")
        return None
    return OpenAI(api_key=api_key)

# Extract text from PDF
def extract_text_from_pdf(uploaded_file) -> str:
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

# Extract text from DOCX
def extract_text_from_docx(uploaded_file) -> str:
    try:
        return docx2txt.process(uploaded_file).strip()
    except Exception as e:
        st.error(f"Error extracting text from DOCX: {str(e)}")
        return ""

# Extract text from file based on type
def extract_text_from_file(uploaded_file) -> str:
    if uploaded_file is None:
        return ""
    fname = uploaded_file.name.lower()
    if fname.endswith(".pdf"):
        return extract_text_from_pdf(uploaded_file)
    elif fname.endswith(".docx") or fname.endswith(".doc"):
        return extract_text_from_docx(uploaded_file)
    else:
        st.error(f"Unsupported file type {fname}. Supported: PDF, DOCX")
        return ""

# Preprocess text: lower, remove special chars & extra spaces
def preprocess_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    return text.strip().lower()

# Extract keywords using TFIDF and NLTK stopwords
def extract_keywords(text: str, top_k: int = 20) -> List[str]:
    try:
        cleaned = preprocess_text(text)
        stop_words = set(stopwords.words('english'))
        words = word_tokenize(cleaned)
        filtered = [w for w in words if w not in stop_words and len(w) > 2]
        if not filtered:
            return []
        vec = TfidfVectorizer(max_features=top_k, stop_words='english')
        tfidf_matrix = vec.fit_transform([' '.join(filtered)])
        feature_names = vec.get_feature_names_out()
        scores = tfidf_matrix.toarray()[0]
        kw_scores = list(zip(feature_names, scores))
        kw_scores.sort(key=lambda x: x[1], reverse=True)
        return [k for k, s in kw_scores if s > 0]
    except Exception as e:
        st.warning(f"Keyword extraction failed: {str(e)}")
        return []

# Extract tech skills (basic example)
def extract_skills(text: str) -> List[str]:
    patterns = [
        r'\b(python|java|javascript|c\+\+|c#|php|ruby|go|swift|kotlin|scala|r)\b',
        r'\b(html|css|react|angular|vue|node\.?js|express|django|flask|spring)\b',
        r'\b(mysql|postgresql|mongodb|redis|elasticsearch|oracle|sql server)\b',
        r'\b(aws|azure|gcp|google cloud|kubernetes|docker|terraform)\b',
        r'\b(machine learning|deep learning|data science|pandas|numpy|tensorflow|pytorch)\b',
        r'\b(git|jenkins|agile|scrum|rest|api|microservices)\b'
    ]
    text_low = text.lower()
    skills = set()
    for pat in patterns:
        skills.update(re.findall(pat, text_low, re.I))
    return list(skills)

# Get embedding from OpenAI
@st.cache_data
def get_embedding(text: str, client) -> Optional[List[float]]:
    if not client or not text.strip():
        return None
    try:
        resp = client.embeddings.create(input=text.replace("\n", " "), model="text-embedding-3-small")
        return resp.data[0].embedding
    except Exception as e:
        st.error(f"OpenAI embedding error: {str(e)}")
        return None

# Cosine similarity
def cosine_sim(v1, v2) -> float:
    if v1 is None or v2 is None:
        return 0.0
    return float(cosine_similarity(np.array(v1).reshape(1, -1), np.array(v2).reshape(1, -1))[0][0])

# Hard match score calculation (keyword & skills overlap)
def calculate_hard_match(resume_text: str, jd_text: str) -> Dict:
    resume_keywords = extract_keywords(resume_text, 50)
    jd_keywords = extract_keywords(jd_text, 50)
    resume_skills = extract_skills(resume_text)
    jd_skills = extract_skills(jd_text)

    kw_overlap = len(set(resume_keywords) & set(jd_keywords))
    kw_score = (kw_overlap / max(len(jd_keywords), 1)) * 100
    skill_overlap = len(set(resume_skills) & set(jd_skills))
    skill_score = (skill_overlap / max(len(jd_skills), 1)) * 100
    missing_skills = list(set(jd_skills) - set(resume_skills))

    hard_score = (kw_score * 0.6 + skill_score * 0.4)
    return {
        "score": min(hard_score, 100),
        "keyword_overlap": kw_overlap,
        "skills_overlap": skill_overlap,
        "missing_skills": missing_skills,
        "resume_skills": resume_skills,
        "jd_skills": jd_skills
    }

# Semantic score
def calculate_semantic(resume_emb, jd_emb):
    sim = cosine_sim(resume_emb, jd_emb)
    return (sim + 1) / 2 * 100  # Normalize from [-1,1] to [0,100]

# Final weighted score
def final_score(hard_score, semantic_score, hard_w=0.4, semantic_w=0.6):
    return min(hard_score * hard_w + semantic_score * semantic_w, 100)

# Verdict based on score
def get_verdict(score):
    if score >= 75:
        return "High Suitability"
    elif score >=50:
        return "Medium Suitability"
    else:
        return "Low Suitability"

# Extract contact info
def extract_contact(text: str) -> Dict:
    contact = {}
    emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)
    phones = re.findall(r'\+?\d[\d\s\-\(\)]{8,}\d', text)
    contact['email'] = emails[0] if emails else ""
    contact['phone'] = phones[0] if phones else ""
    return contact

# Suggestions
def generate_suggestions(hard_res, semantic_score):
    suggestions = []
    missing_skills = hard_res.get('missing_skills', [])
    if missing_skills:
        suggestions.append(f"Consider adding these missing skills: {', '.join(missing_skills[:5])}")
    if hard_res['score'] < 50:
        suggestions.append("Add more relevant keywords from the job description to your resume.")
        suggestions.append("Highlight specific projects showing required skills.")
    if semantic_score < 50:
        suggestions.append("Restructure your resume to better align with job requirements.")
        suggestions.append("Use more industry-specific terminology.")
    if hard_res['skills_overlap'] < 3:
        suggestions.append("Add more technical skills matching job needs.")
    suggestions.append("Quantify your achievements with metrics/results.")
    suggestions.append("Include relevant certifications or training.")
    return suggestions

# Evaluate a single resume
def evaluate_resume(resume_text, jd_text, client):
    resume_emb = get_embedding(resume_text, client)
    jd_emb = get_embedding(jd_text, client)
    hard_res = calculate_hard_match(resume_text, jd_text)
    semantic_score = calculate_semantic(resume_emb, jd_emb)
    final = final_score(hard_res['score'], semantic_score)
    verdict = get_verdict(final)
    contact_info = extract_contact(resume_text)
    suggestions = generate_suggestions(hard_res, semantic_score)
    return {
        "final_score": final,
        "hard_score": hard_res['score'],
        "semantic_score": semantic_score,
        "verdict": verdict,
        "contact_info": contact_info,
        "skills_analysis": {
            "resume_skills": hard_res['resume_skills'],
            "jd_skills": hard_res['jd_skills'],
            "missing_skills": hard_res['missing_skills'],
            "skills_overlap": hard_res['skills_overlap']
        },
        "suggestions": suggestions,
        "timestamp": datetime.now().isoformat()
    }

def get_verdict_color(verdict):
    colors = {
        "High Suitability": "green",
        "Medium Suitability": "orange",
        "Low Suitability": "red"
    }
    return colors.get(verdict, "gray")

# Streamlit UI with enhanced theming and layout
def main():
    st.set_page_config(
        page_title="Automated Resume Relevance Check System",
        page_icon="📄",
        layout="wide"
    )
    # Custom CSS for professional look
    custom_css = """
    <style>
    .block-container {
        max-width: 1000px;
        padding: 2rem 2rem 2rem 2rem;
    }
    .stButton>button {
        background-color: #4B8BBE;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 10px 20px;
        border: none;
        transition: background-color 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #306998;
        color: white;
    }
    header .css-1v3fvcr {
        background-color: #14213d !important;
        color: white !important;
    }
    footer {
        visibility: hidden;
    }
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)

    st.title("🎯 Automated Resume Relevance Check System")
    st.markdown("#### Innomatics Research Labs - AI Powered Resume Evaluation")

    client = init_openai_client()
    if not client:
        return

    # Sidebar navigation
    page = st.sidebar.selectbox("Navigate", ["Resume Evaluation", "Dashboard"])

    # Session state init
    if "evaluations" not in st.session_state:
        st.session_state.evaluations = []

    if page == "Resume Evaluation":
        st.header("📄 Resume Evaluation")

        # Job description input
        col1, col2 = st.columns([3,1])
        with col1:
            st.subheader("1. Job Description Input")
            jd_textarea = st.text_area("Paste Job Description (Optional if uploading file):", height=180)
            jd_file = st.file_uploader("Or Upload Job Description file (PDF/DOCX):", type=['pdf','docx'], help="Upload job description file in PDF or DOCX format")
            job_desc_text = ""
            if jd_file is not None:
                job_desc_text = extract_text_from_file(jd_file)
                if not job_desc_text:
                    st.warning("Uploaded Job description file is empty or error extracting text")
            elif jd_textarea.strip():
                job_desc_text = jd_textarea.strip()
            else:
                st.warning("Please enter or upload a job description.")
                st.stop()

        with col2:
            st.subheader("Job ID (Optional)")
            job_id = st.text_input("Enter Job ID:", placeholder="e.g. JOB-2025-001")

        # Resume upload
        st.subheader("2. Upload Resumes (PDF or DOCX, multiple allowed)")
        uploaded_resumes = st.file_uploader("Upload resume files:", type=['pdf','docx'], accept_multiple_files=True)

        st.subheader("3. Evaluation Settings")
        hard_w = st.slider("Hard Match Weight (Keywords & Skills)", 0.0, 1.0, 0.4, 0.1)
        semantic_w = st.slider("Semantic Match Weight (Embedding Similarity)", 0.0, 1.0, 0.6, 0.1)

        if st.button("🚀 Evaluate Resumes"):
            if not uploaded_resumes:
                st.error("Please upload at least one resume file")
                return
            if not job_desc_text.strip():
                st.error("Job description is required")
                return
            st.info(f"Evaluating {len(uploaded_resumes)} resumes...")

            progress_bar = st.progress(0)
            results = []

            for i, uf in enumerate(uploaded_resumes):
                st.write(f"Processing {uf.name}...")
                resume_text = extract_text_from_file(uf)
                if not resume_text:
                    st.warning(f"Could not extract text from {uf.name}")
                    continue
                res_eval = evaluate_resume(resume_text, job_desc_text, client)
                res_eval['filename'] = uf.name
                res_eval['job_id'] = job_id
                results.append(res_eval)
                st.session_state.evaluations.append(res_eval)
                progress_bar.progress((i+1)/len(uploaded_resumes))

            st.success("Evaluation complete!")

            if results:
                # Show results table with color coded verdict
                df_res = pd.DataFrame([{
                    "Resume": r["filename"],
                    "Final Score": round(r["final_score"],1),
                    "Hard Match": round(r["hard_score"],1),
                    "Semantic Match": round(r["semantic_score"],1),
                    "Verdict": r["verdict"],
                    "Missing Skills": len(r["skills_analysis"]["missing_skills"]),
                    "Timestamp": r["timestamp"].split("T")[0]
                } for r in results])

                def color_verdict(val):
                    color = get_verdict_color(val)
                    return f"background-color: {color}; color: white; font-weight: bold"
                st.subheader("📊 Evaluation Summary")
                st.dataframe(df_res.style.applymap(color_verdict, subset=["Verdict"]), use_container_width=True)

                # Detailed per resume
                for r in results:
                    st.markdown(f"---\n### Detailed Analysis: {r['filename']}")
                    st.metric("Final Relevance Score", f"{r['final_score']:.1f}/100")
                    st.metric("Hard Match Score", f"{r['hard_score']:.1f}/100")
                    st.metric("Semantic Match Score", f"{r['semantic_score']:.1f}/100")
                    st.markdown(f"**Verdict:** <span style='color:{get_verdict_color(r['verdict'])}; font-weight:bold'>{r['verdict']}</span>", unsafe_allow_html=True)
                    with st.expander("Skills Analysis"):
                        st.write("Resume Skills:")
                        if r['skills_analysis']['resume_skills']:
                            for skill in r['skills_analysis']['resume_skills']:
                                st.write(f"• {skill}")
                        else:
                            st.write("_No technical skills identified_")
                        st.write("Required Skills:")
                        if r['skills_analysis']['jd_skills']:
                            for skill in r['skills_analysis']['jd_skills']:
                                mark = "✅" if skill in r['skills_analysis']['resume_skills'] else "❌"
                                st.write(f"{mark} {skill}")
                        else:
                            st.write("_No specific skills identified in job description_")
                    with st.expander("Missing Skills"):
                        if r['skills_analysis']['missing_skills']:
                            for ms in r['skills_analysis']['missing_skills']:
                                st.write(f"• {ms}")
                        else:
                            st.write("_No missing skills identified_")
                    with st.expander("Contact Information"):
                        email = r['contact_info'].get('email', '')
                        phone = r['contact_info'].get('phone', '')
                        if email: st.write(f"Email: {email}")
                        if phone: st.write(f"Phone: {phone}")
                        if not email and not phone:
                            st.write("_No contact information found_")
                    with st.expander("Improvement Suggestions"):
                        for idx, sug in enumerate(r['suggestions'], 1):
                            st.write(f"{idx}. {sug}")

    else:  # Dashboard
        st.header("📈 Evaluation Dashboard")
        if not st.session_state.evaluations:
            st.info("No evaluations done yet. Go to the Resume Evaluation page to start.")
            return

        evaluations = st.session_state.evaluations

        # Filters
        verdict_filter = st.selectbox("Filter by Suitability Verdict", ["All", "High Suitability", "Medium Suitability", "Low Suitability"])
        min_score = st.slider("Minimum Final Score", 0, 100, 0)
        job_ids = list(sorted(set(e.get("job_id") for e in evaluations if e.get("job_id"))))
        job_filter = st.selectbox("Filter by Job ID", ["All"] + job_ids) if job_ids else None

        filtered = evaluations
        if verdict_filter != "All":
            filtered = [e for e in filtered if e['verdict'] == verdict_filter]
        filtered = [e for e in filtered if e['final_score'] >= min_score]
        if job_filter and job_filter != "All":
            filtered = [e for e in filtered if e.get('job_id') == job_filter]

        if not filtered:
            st.info("No results match the current filters.")
            return

        df = pd.DataFrame([{
            "Resume": r["filename"],
            "Job ID": r.get("job_id", ""),
            "Final Score": round(r["final_score"],1),
            "Hard Match": round(r["hard_score"],1),
            "Semantic Match": round(r["semantic_score"],1),
            "Verdict": r["verdict"],
            "Missing Skills": len(r["skills_analysis"]["missing_skills"]),
            "Timestamp": r["timestamp"].split("T")[0]
        } for r in filtered])

        def color_verdict(val):
            color = get_verdict_color(val)
            return f"background-color: {color}; color: white; font-weight: bold"

        st.dataframe(df.style.applymap(color_verdict, subset=["Verdict"]), use_container_width=True)

        if st.button("Export Filtered Results as CSV"):
            csv_data = df.to_csv(index=False)
            st.download_button(label="Download CSV", data=csv_data, file_name=f"resume_evaluations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", mime='text/csv')

if __name__ == "__main__":
    main()
