
import streamlit as st
import pandas as pd
import numpy as np
import re
import json
import io
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import openai
from openai import OpenAI

# File processing libraries
import PyPDF2
from docx import Document as DocxDocument
import docx2txt

# ML and similarity libraries
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download required NLTK data (run once)
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
except:
    pass

# Configuration
st.set_page_config(
    page_title="Resume Relevance Check System",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize OpenAI client (you need to set your API key)
@st.cache_resource
def init_openai_client():
    """Initialize OpenAI client with API key from secrets or environment"""
    try:
        # Try to get API key from Streamlit secrets
        api_key = st.secrets.get("OPENAI_API_KEY", None)
        if not api_key:
            # Fallback to environment variable or user input
            import os
            api_key = os.getenv("OPENAI_API_KEY")

        if api_key:
            return OpenAI(api_key=api_key)
        else:
            st.error("OpenAI API key not found. Please set it in secrets.toml or environment variables.")
            return None
    except Exception as e:
        st.error(f"Error initializing OpenAI client: {str(e)}")
        return None

# Text extraction functions
def extract_text_from_pdf(uploaded_file) -> str:
    """Extract text from PDF file"""
    try:
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text.strip()
    except Exception as e:
        st.error(f"Error extracting text from PDF: {str(e)}")
        return ""

def extract_text_from_docx(uploaded_file) -> str:
    """Extract text from DOCX file"""
    try:
        # Save uploaded file to temporary location and extract text
        text = docx2txt.process(uploaded_file)
        return text.strip()
    except Exception as e:
        st.error(f"Error extracting text from DOCX: {str(e)}")
        return ""

def extract_text_from_file(uploaded_file) -> str:
    """Extract text from uploaded file based on file type"""
    if uploaded_file is None:
        return ""

    file_extension = uploaded_file.name.lower().split('.')[-1]

    if file_extension == 'pdf':
        return extract_text_from_pdf(uploaded_file)
    elif file_extension in ['docx', 'doc']:
        return extract_text_from_docx(uploaded_file)
    else:
        st.error(f"Unsupported file type: {file_extension}")
        return ""

# Text preprocessing functions
def preprocess_text(text: str) -> str:
    """Clean and preprocess text"""
    # Remove extra whitespace and normalize
    text = re.sub(r'\s+', ' ', text)
    # Remove special characters but keep alphanumeric and spaces
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    return text.strip().lower()

def extract_keywords(text: str, top_k: int = 20) -> List[str]:
    """Extract top keywords from text using TF-IDF"""
    try:
        # Preprocess text
        cleaned_text = preprocess_text(text)

        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        words = word_tokenize(cleaned_text)
        filtered_words = [word for word in words if word not in stop_words and len(word) > 2]

        if not filtered_words:
            return []

        # Create TF-IDF vectorizer
        vectorizer = TfidfVectorizer(max_features=top_k, stop_words='english')
        tfidf_matrix = vectorizer.fit_transform([' '.join(filtered_words)])

        # Get feature names (keywords)
        feature_names = vectorizer.get_feature_names_out()
        tfidf_scores = tfidf_matrix.toarray()[0]

        # Get top keywords with scores
        keyword_scores = list(zip(feature_names, tfidf_scores))
        keyword_scores.sort(key=lambda x: x[1], reverse=True)

        return [keyword for keyword, score in keyword_scores if score > 0]
    except Exception as e:
        st.warning(f"Error extracting keywords: {str(e)}")
        return []

# Information extraction functions
def extract_contact_info(text: str) -> Dict[str, str]:
    """Extract contact information from text"""
    contact_info = {}

    # Email extraction
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    emails = re.findall(email_pattern, text)
    contact_info['email'] = emails[0] if emails else ""

    # Phone extraction
    phone_pattern = r'\+?\d[\d\s\-\(\)]{8,}\d'
    phones = re.findall(phone_pattern, text)
    contact_info['phone'] = phones[0] if phones else ""

    return contact_info

def extract_skills(text: str) -> List[str]:
    """Extract skills from text based on common skill keywords"""
    # Common technical skills (can be expanded)
    skill_patterns = [
        # Programming languages
        r'\b(python|java|javascript|c\+\+|c#|php|ruby|go|swift|kotlin|scala|r)\b',
        # Web technologies
        r'\b(html|css|react|angular|vue|node\.?js|express|django|flask|spring)\b',
        # Databases
        r'\b(mysql|postgresql|mongodb|redis|elasticsearch|oracle|sql server)\b',
        # Cloud platforms
        r'\b(aws|azure|gcp|google cloud|kubernetes|docker|terraform)\b',
        # Data science
        r'\b(machine learning|deep learning|data science|pandas|numpy|tensorflow|pytorch)\b',
        # Others
        r'\b(git|jenkins|agile|scrum|rest|api|microservices)\b'
    ]

    skills = set()
    text_lower = text.lower()

    for pattern in skill_patterns:
        matches = re.findall(pattern, text_lower, re.IGNORECASE)
        skills.update(matches)

    return list(skills)

# OpenAI embedding functions
@st.cache_data
def get_embedding(text: str, _client) -> Optional[List[float]]:
    """Get embedding for text using OpenAI API"""
    if not _client:
        return None

    try:
        # Clean text
        text = text.replace("\n", " ").strip()
        if not text:
            return None

        response = _client.embeddings.create(
            input=text,
            model="text-embedding-3-small"
        )
        return response.data[0].embedding
    except Exception as e:
        st.error(f"Error getting embedding: {str(e)}")
        return None

def calculate_cosine_similarity(embedding1: List[float], embedding2: List[float]) -> float:
    """Calculate cosine similarity between two embeddings"""
    if not embedding1 or not embedding2:
        return 0.0

    # Convert to numpy arrays
    vec1 = np.array(embedding1).reshape(1, -1)
    vec2 = np.array(embedding2).reshape(1, -1)

    # Calculate cosine similarity
    similarity = cosine_similarity(vec1, vec2)[0][0]
    return float(similarity)

# Scoring functions
def calculate_hard_match_score(resume_text: str, job_description: str) -> Dict:
    """Calculate hard match score based on keyword overlap"""
    resume_keywords = extract_keywords(resume_text, top_k=50)
    jd_keywords = extract_keywords(job_description, top_k=50)

    resume_skills = extract_skills(resume_text)
    jd_skills = extract_skills(job_description)

    # Calculate keyword overlap
    keyword_overlap = len(set(resume_keywords) & set(jd_keywords))
    keyword_score = (keyword_overlap / max(len(jd_keywords), 1)) * 100

    # Calculate skills overlap
    skills_overlap = len(set(resume_skills) & set(jd_skills))
    skills_score = (skills_overlap / max(len(jd_skills), 1)) * 100

    # Missing skills
    missing_skills = list(set(jd_skills) - set(resume_skills))

    # Combined hard match score (weighted average)
    hard_score = (keyword_score * 0.6 + skills_score * 0.4)

    return {
        'score': min(hard_score, 100),
        'keyword_overlap': keyword_overlap,
        'skills_overlap': skills_overlap,
        'missing_skills': missing_skills,
        'resume_skills': resume_skills,
        'jd_skills': jd_skills
    }

def calculate_semantic_score(resume_embedding: List[float], jd_embedding: List[float]) -> float:
    """Calculate semantic similarity score"""
    if not resume_embedding or not jd_embedding:
        return 0.0

    similarity = calculate_cosine_similarity(resume_embedding, jd_embedding)
    # Convert to 0-100 scale
    return (similarity + 1) / 2 * 100  # Normalize from [-1,1] to [0,100]

def calculate_final_score(hard_score: float, semantic_score: float, 
                         hard_weight: float = 0.4, semantic_weight: float = 0.6) -> float:
    """Calculate final weighted score"""
    final_score = (hard_score * hard_weight) + (semantic_score * semantic_weight)
    return min(final_score, 100)

def get_verdict(score: float) -> str:
    """Get verdict based on score"""
    if score >= 75:
        return "High Suitability"
    elif score >= 50:
        return "Medium Suitability"
    else:
        return "Low Suitability"

def get_verdict_color(verdict: str) -> str:
    """Get color for verdict display"""
    colors = {
        "High Suitability": "green",
        "Medium Suitability": "orange",
        "Low Suitability": "red"
    }
    return colors.get(verdict, "gray")

# Main evaluation function
def evaluate_resume(resume_text: str, job_description: str, client) -> Dict:
    """Main function to evaluate resume against job description"""

    # Get embeddings
    resume_embedding = get_embedding(resume_text, client)
    jd_embedding = get_embedding(job_description, client)

    # Calculate scores
    hard_match_result = calculate_hard_match_score(resume_text, job_description)
    semantic_score = calculate_semantic_score(resume_embedding, jd_embedding)

    # Calculate final score
    final_score = calculate_final_score(hard_match_result['score'], semantic_score)

    # Get verdict
    verdict = get_verdict(final_score)

    # Extract contact info
    contact_info = extract_contact_info(resume_text)

    # Generate suggestions
    suggestions = generate_suggestions(hard_match_result, semantic_score)

    return {
        'final_score': final_score,
        'hard_score': hard_match_result['score'],
        'semantic_score': semantic_score,
        'verdict': verdict,
        'contact_info': contact_info,
        'skills_analysis': {
            'resume_skills': hard_match_result['resume_skills'],
            'jd_skills': hard_match_result['jd_skills'],
            'missing_skills': hard_match_result['missing_skills'],
            'skills_overlap': hard_match_result['skills_overlap']
        },
        'keyword_analysis': {
            'keyword_overlap': hard_match_result['keyword_overlap']
        },
        'suggestions': suggestions,
        'timestamp': datetime.now().isoformat()
    }

def generate_suggestions(hard_match_result: Dict, semantic_score: float) -> List[str]:
    """Generate improvement suggestions"""
    suggestions = []

    missing_skills = hard_match_result.get('missing_skills', [])

    if missing_skills:
        suggestions.append(f"Consider adding these missing skills: {', '.join(missing_skills[:5])}")

    if hard_match_result['score'] < 50:
        suggestions.append("Add more relevant keywords from the job description to your resume")
        suggestions.append("Highlight specific projects that demonstrate required skills")

    if semantic_score < 50:
        suggestions.append("Restructure your resume content to better align with job requirements")
        suggestions.append("Use more industry-specific terminology and phrases")

    if hard_match_result['skills_overlap'] < 3:
        suggestions.append("Add more technical skills that match the job requirements")

    suggestions.append("Quantify your achievements with specific metrics and results")
    suggestions.append("Include relevant certifications or training programs")

    return suggestions

# Initialize session state
def init_session_state():
    """Initialize session state variables"""
    if 'evaluations' not in st.session_state:
        st.session_state.evaluations = []
    if 'current_job_id' not in st.session_state:
        st.session_state.current_job_id = None

# Streamlit UI
def main():
    init_session_state()

    st.title("🎯 Automated Resume Relevance Check System")
    st.markdown("**Innomatics Research Labs - AI-Powered Resume Evaluation**")

    # Initialize OpenAI client
    client = init_openai_client()

    if not client:
        st.stop()

    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page",
        ["Resume Evaluation", "Dashboard", "Analytics"]
    )

    if page == "Resume Evaluation":
        show_evaluation_page(client)
    elif page == "Dashboard":
        show_dashboard_page()
    else:
        show_analytics_page()

def show_evaluation_page(client):
    """Show the main evaluation page"""
    st.header("📄 Resume Evaluation")

    # Job description input
    st.subheader("1. Job Description")
    job_description = st.text_area(
        "Paste the job description here:",
        height=200,
        placeholder="Paste the complete job description including required skills, qualifications, and responsibilities..."
    )

    # Job ID input
    job_id = st.text_input(
        "Job ID (optional):",
        placeholder="e.g., JOB-2024-001"
    )

    # Resume upload
    st.subheader("2. Resume Upload")
    uploaded_files = st.file_uploader(
        "Upload resume files (PDF or DOCX):",
        type=['pdf', 'docx'],
        accept_multiple_files=True,
        help="You can upload multiple resumes for batch processing"
    )

    # Evaluation settings
    st.subheader("3. Evaluation Settings")
    col1, col2 = st.columns(2)

    with col1:
        hard_weight = st.slider(
            "Hard Match Weight:",
            min_value=0.0,
            max_value=1.0,
            value=0.4,
            step=0.1,
            help="Weight for keyword and skill matching"
        )

    with col2:
        semantic_weight = st.slider(
            "Semantic Match Weight:",
            min_value=0.0,
            max_value=1.0,
            value=0.6,
            step=0.1,
            help="Weight for semantic similarity"
        )

    # Evaluate button
    if st.button("🚀 Evaluate Resumes", type="primary"):
        if not job_description.strip():
            st.error("Please provide a job description")
            return

        if not uploaded_files:
            st.error("Please upload at least one resume")
            return

        # Process each resume
        progress_bar = st.progress(0)
        results = []

        for i, uploaded_file in enumerate(uploaded_files):
            st.write(f"Processing: {uploaded_file.name}")

            # Extract text
            resume_text = extract_text_from_file(uploaded_file)

            if resume_text:
                # Evaluate resume
                evaluation_result = evaluate_resume(resume_text, job_description, client)
                evaluation_result['filename'] = uploaded_file.name
                evaluation_result['job_id'] = job_id
                evaluation_result['job_description'] = job_description[:200] + "..." if len(job_description) > 200 else job_description

                results.append(evaluation_result)

                # Store in session state
                st.session_state.evaluations.append(evaluation_result)

            progress_bar.progress((i + 1) / len(uploaded_files))

        # Display results
        st.success(f"Evaluated {len(results)} resumes successfully!")

        # Show results table
        display_results_table(results)

        # Show detailed results
        for result in results:
            display_detailed_result(result)

def display_results_table(results: List[Dict]):
    """Display results in a table format"""
    if not results:
        return

    st.subheader("📊 Evaluation Results Summary")

    # Create DataFrame
    df_data = []
    for result in results:
        df_data.append({
            'Resume': result['filename'],
            'Final Score': f"{result['final_score']:.1f}",
            'Hard Match': f"{result['hard_score']:.1f}",
            'Semantic Match': f"{result['semantic_score']:.1f}",
            'Verdict': result['verdict'],
            'Missing Skills': len(result['skills_analysis']['missing_skills']),
            'Timestamp': result['timestamp'].split('T')[0]  # Just date
        })

    df = pd.DataFrame(df_data)

    # Style the dataframe
    def color_verdict(val):
        color = get_verdict_color(val)
        return f'background-color: {color}; color: white; font-weight: bold'

    styled_df = df.style.applymap(color_verdict, subset=['Verdict'])
    st.dataframe(styled_df, use_container_width=True)

def display_detailed_result(result: Dict):
    """Display detailed result for a single resume"""
    st.subheader(f"📋 Detailed Analysis: {result['filename']}")

    # Score overview
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Final Score",
            f"{result['final_score']:.1f}/100",
            delta=None
        )

    with col2:
        st.metric(
            "Hard Match",
            f"{result['hard_score']:.1f}/100"
        )

    with col3:
        st.metric(
            "Semantic Match",
            f"{result['semantic_score']:.1f}/100"
        )

    with col4:
        verdict_color = get_verdict_color(result['verdict'])
        st.markdown(f"**Verdict:** :{verdict_color}[{result['verdict']}]")

    # Skills analysis
    with st.expander("🛠️ Skills Analysis"):
        skills_data = result['skills_analysis']

        col1, col2 = st.columns(2)

        with col1:
            st.write("**Resume Skills:**")
            if skills_data['resume_skills']:
                for skill in skills_data['resume_skills']:
                    st.write(f"• {skill}")
            else:
                st.write("No technical skills identified")

        with col2:
            st.write("**Required Skills:**")
            if skills_data['jd_skills']:
                for skill in skills_data['jd_skills']:
                    if skill in skills_data['resume_skills']:
                        st.write(f"✅ {skill}")
                    else:
                        st.write(f"❌ {skill}")
            else:
                st.write("No specific skills identified in job description")

    # Missing skills
    if result['skills_analysis']['missing_skills']:
        with st.expander("❗ Missing Skills"):
            for skill in result['skills_analysis']['missing_skills']:
                st.write(f"• {skill}")

    # Contact information
    if any(result['contact_info'].values()):
        with st.expander("📞 Contact Information"):
            if result['contact_info']['email']:
                st.write(f"**Email:** {result['contact_info']['email']}")
            if result['contact_info']['phone']:
                st.write(f"**Phone:** {result['contact_info']['phone']}")

    # Suggestions
    with st.expander("💡 Improvement Suggestions"):
        for i, suggestion in enumerate(result['suggestions'], 1):
            st.write(f"{i}. {suggestion}")

    st.divider()

def show_dashboard_page():
    """Show the dashboard page"""
    st.header("📈 Results Dashboard")

    if not st.session_state.evaluations:
        st.info("No evaluations yet. Go to the Resume Evaluation page to get started.")
        return

    # Summary statistics
    evaluations = st.session_state.evaluations

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Evaluations", len(evaluations))

    with col2:
        avg_score = np.mean([e['final_score'] for e in evaluations])
        st.metric("Average Score", f"{avg_score:.1f}")

    with col3:
        high_suitable = len([e for e in evaluations if e['verdict'] == 'High Suitability'])
        st.metric("High Suitability", high_suitable)

    with col4:
        recent_date = max([e['timestamp'].split('T')[0] for e in evaluations])
        st.metric("Last Evaluation", recent_date)

    # Filters
    st.subheader("🔍 Filters")
    col1, col2, col3 = st.columns(3)

    with col1:
        verdict_filter = st.selectbox(
            "Filter by Verdict:",
            ["All", "High Suitability", "Medium Suitability", "Low Suitability"]
        )

    with col2:
        min_score = st.number_input("Minimum Score:", min_value=0, max_value=100, value=0)

    with col3:
        job_ids = list(set([e.get('job_id', 'Unknown') for e in evaluations if e.get('job_id')]))
        if job_ids:
            job_filter = st.selectbox("Filter by Job ID:", ["All"] + job_ids)
        else:
            job_filter = "All"

    # Apply filters
    filtered_evaluations = evaluations.copy()

    if verdict_filter != "All":
        filtered_evaluations = [e for e in filtered_evaluations if e['verdict'] == verdict_filter]

    filtered_evaluations = [e for e in filtered_evaluations if e['final_score'] >= min_score]

    if job_filter != "All":
        filtered_evaluations = [e for e in filtered_evaluations if e.get('job_id') == job_filter]

    # Results table
    st.subheader(f"📊 Results ({len(filtered_evaluations)} items)")

    if filtered_evaluations:
        display_results_table(filtered_evaluations)

        # Export functionality
        if st.button("📥 Export Results to CSV"):
            df_data = []
            for result in filtered_evaluations:
                df_data.append({
                    'Filename': result['filename'],
                    'Job_ID': result.get('job_id', ''),
                    'Final_Score': result['final_score'],
                    'Hard_Score': result['hard_score'],
                    'Semantic_Score': result['semantic_score'],
                    'Verdict': result['verdict'],
                    'Email': result['contact_info'].get('email', ''),
                    'Phone': result['contact_info'].get('phone', ''),
                    'Resume_Skills': ', '.join(result['skills_analysis']['resume_skills']),
                    'Missing_Skills': ', '.join(result['skills_analysis']['missing_skills']),
                    'Timestamp': result['timestamp']
                })

            df = pd.DataFrame(df_data)
            csv = df.to_csv(index=False)

            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"resume_evaluations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    else:
        st.info("No results match the current filters.")

def show_analytics_page():
    """Show analytics page"""
    st.header("📊 Analytics")

    if not st.session_state.evaluations:
        st.info("No data available for analytics. Evaluate some resumes first.")
        return

    evaluations = st.session_state.evaluations

    # Score distribution
    st.subheader("Score Distribution")
    scores = [e['final_score'] for e in evaluations]

    # Create score ranges
    score_ranges = ['0-25', '26-50', '51-75', '76-100']
    score_counts = [
        len([s for s in scores if 0 <= s <= 25]),
        len([s for s in scores if 26 <= s <= 50]),
        len([s for s in scores if 51 <= s <= 75]),
        len([s for s in scores if 76 <= s <= 100])
    ]

    col1, col2 = st.columns(2)

    with col1:
        st.bar_chart(dict(zip(score_ranges, score_counts)))

    with col2:
        verdict_counts = {}
        for e in evaluations:
            verdict = e['verdict']
            verdict_counts[verdict] = verdict_counts.get(verdict, 0) + 1

        st.bar_chart(verdict_counts)

    # Skills analysis
    st.subheader("Skills Analysis")

    all_missing_skills = []
    all_resume_skills = []

    for e in evaluations:
        all_missing_skills.extend(e['skills_analysis']['missing_skills'])
        all_resume_skills.extend(e['skills_analysis']['resume_skills'])

    col1, col2 = st.columns(2)

    with col1:
        st.write("**Most Common Missing Skills:**")
        if all_missing_skills:
            missing_skills_counts = {}
            for skill in all_missing_skills:
                missing_skills_counts[skill] = missing_skills_counts.get(skill, 0) + 1

            # Sort by frequency
            sorted_missing = sorted(missing_skills_counts.items(), key=lambda x: x[1], reverse=True)[:10]
            for skill, count in sorted_missing:
                st.write(f"• {skill} ({count} times)")
        else:
            st.write("No missing skills data available")

    with col2:
        st.write("**Most Common Resume Skills:**")
        if all_resume_skills:
            resume_skills_counts = {}
            for skill in all_resume_skills:
                resume_skills_counts[skill] = resume_skills_counts.get(skill, 0) + 1

            # Sort by frequency
            sorted_resume = sorted(resume_skills_counts.items(), key=lambda x: x[1], reverse=True)[:10]
            for skill, count in sorted_resume:
                st.write(f"• {skill} ({count} times)")
        else:
            st.write("No resume skills data available")

if __name__ == "__main__":
    main()
