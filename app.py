import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import tempfile
import os
import json
import logging
import sys

# Add current directory to Python path to resolve import issues
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# Import custom modules with error handling
try:
    from database import DatabaseManager
    from resume_parser import ResumeParser
    from job_parser import JobDescriptionParser
    from matching_engine import MatchingEngine
except ImportError as e:
    st.error(f"Import Error: {e}")
    st.error("Please make sure all Python files are in the same directory")
    st.stop()

# Configure logging
logging.basicConfig(level=logging.INFO)

# Page configuration
st.set_page_config(
    page_title="Innomatics Resume Relevance Check System",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1f4e79 0%, #2d5aa0 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1f4e79;
    }
    .verdict-high {
        background-color: #d4edda;
        color: #155724;
        padding: 10px;
        border-radius: 5px;
        border-left: 4px solid #28a745;
    }
    .verdict-medium {
        background-color: #fff3cd;
        color: #856404;
        padding: 10px;
        border-radius: 5px;
        border-left: 4px solid #ffc107;
    }
    .verdict-low {
        background-color: #f8d7da;
        color: #721c24;
        padding: 10px;
        border-radius: 5px;
        border-left: 4px solid #dc3545;
    }
</style>
""", unsafe_allow_html=True)

# Initialize components
@st.cache_resource
def init_components():
    """Initialize all components"""
    db = DatabaseManager()
    parser = ResumeParser()
    job_parser = JobDescriptionParser()
    matcher = MatchingEngine()
    return db, parser, job_parser, matcher

# Page header
st.markdown("""
<div class="main-header">
    <h1>🎯 Innomatics Resume Relevance Check System</h1>
    <p>Automated AI-powered resume evaluation and job matching system</p>
</div>
""", unsafe_allow_html=True)

# Initialize components
db, resume_parser, job_parser, matching_engine = init_components()

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox(
    "Choose a page",
    ["📊 Dashboard", "📝 Upload Job Description", "📄 Upload Resume", "🔍 Evaluate Resumes", "📈 Analytics"]
)

if page == "📊 Dashboard":
    st.header("System Overview")
    
    # Get statistics
    stats = db.get_evaluation_stats()
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(
            f"""<div class="metric-card">
                <h3>{stats['total_jobs']}</h3>
                <p>Total Job Descriptions</p>
            </div>""", 
            unsafe_allow_html=True
        )
    
    with col2:
        st.markdown(
            f"""<div class="metric-card">
                <h3>{stats['total_resumes']}</h3>
                <p>Total Resumes</p>
            </div>""", 
            unsafe_allow_html=True
        )
    
    with col3:
        st.markdown(
            f"""<div class="metric-card">
                <h3>{stats['total_evaluations']}</h3>
                <p>Total Evaluations</p>
            </div>""", 
            unsafe_allow_html=True
        )
    
    with col4:
        verdict_dist = stats.get('verdict_distribution', {})
        high_percentage = round((verdict_dist.get('High', 0) / max(stats['total_evaluations'], 1)) * 100, 1)
        st.markdown(
            f"""<div class="metric-card">
                <h3>{high_percentage}%</h3>
                <p>High Match Rate</p>
            </div>""", 
            unsafe_allow_html=True
        )
    
    # Verdict distribution chart
    if stats['verdict_distribution']:
        st.subheader("Verdict Distribution")
        verdict_df = pd.DataFrame(list(stats['verdict_distribution'].items()), columns=['Verdict', 'Count'])
        fig = px.pie(verdict_df, values='Count', names='Verdict', 
                    color_discrete_map={'High': '#28a745', 'Medium': '#ffc107', 'Low': '#dc3545'})
        st.plotly_chart(fig, use_container_width=True)
    
    # Recent job descriptions
    st.subheader("Recent Job Descriptions")
    jobs_df = db.get_job_descriptions()
    if not jobs_df.empty:
        st.dataframe(jobs_df[['title', 'company', 'location', 'created_at']].head(10), use_container_width=True)
    else:
        st.info("No job descriptions uploaded yet.")

elif page == "📝 Upload Job Description":
    st.header("Upload Job Description")
    
    with st.form("job_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            title = st.text_input("Job Title*", placeholder="e.g., Senior Python Developer")
            company = st.text_input("Company", placeholder="e.g., Innomatics Research Labs")
            location = st.text_input("Location", placeholder="e.g., Hyderabad, India")
        
        with col2:
            st.markdown("### Skills Requirements")
            must_have_skills = st.text_area("Must-have Skills (comma-separated)*", 
                                           placeholder="Python, Django, SQL, REST API")
            good_to_have_skills = st.text_area("Good-to-have Skills (comma-separated)", 
                                             placeholder="AWS, Docker, React, Machine Learning")
        
        description = st.text_area("Full Job Description*", 
                                  placeholder="Paste the complete job description here...", 
                                  height=300)
        
        submitted = st.form_submit_button("Upload Job Description")
        
        if submitted:
            if title and description and must_have_skills:
                # Process skills
                must_have_list = [skill.strip() for skill in must_have_skills.split(',')]
                good_to_have_list = [skill.strip() for skill in good_to_have_skills.split(',') if skill.strip()]
                
                # Save to database
                job_id = db.add_job_description(
                    title=title,
                    company=company,
                    description=description,
                    must_have_skills=must_have_list,
                    good_to_have_skills=good_to_have_list,
                    location=location
                )
                
                st.success(f"✅ Job description uploaded successfully! (ID: {job_id})")
                st.balloons()
            else:
                st.error("Please fill in all required fields marked with *")

elif page == "📄 Upload Resume":
    st.header("Upload Resume")
    
    uploaded_file = st.file_uploader(
        "Choose a resume file", 
        type=['pdf', 'docx', 'doc'],
        help="Supported formats: PDF, DOCX, DOC"
    )
    
    if uploaded_file is not None:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        
        try:
            st.info("🔄 Processing resume...")
            
            # Parse resume
            file_extension = uploaded_file.name.split('.')[-1]
            parsed_data = resume_parser.parse_resume(tmp_file_path, file_extension)
            
            if parsed_data:
                # Display extracted information
                st.success("✅ Resume parsed successfully!")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Contact Information")
                    st.write(f"**Name:** {parsed_data.get('name', 'Not found')}")
                    st.write(f"**Email:** {parsed_data.get('email', 'Not found')}")
                    st.write(f"**Phone:** {parsed_data.get('phone', 'Not found')}")
                
                with col2:
                    st.subheader("Extracted Skills")
                    skills = parsed_data.get('skills', [])
                    if skills:
                        st.write(", ".join(skills[:10]))  # Show first 10 skills
                    else:
                        st.write("No skills detected")
                
                # Allow manual corrections
                with st.expander("Edit Information (Optional)"):
                    name = st.text_input("Name", value=parsed_data.get('name', ''))
                    email = st.text_input("Email", value=parsed_data.get('email', ''))
                    phone = st.text_input("Phone", value=parsed_data.get('phone', ''))
                    additional_skills = st.text_input("Additional Skills (comma-separated)", 
                                                    placeholder="Add any missing skills...")
                
                if st.button("Save Resume"):
                    # Update with manual corrections
                    final_skills = skills.copy()
                    if additional_skills:
                        final_skills.extend([s.strip() for s in additional_skills.split(',')])
                    
                    # Save to database
                    resume_id = db.add_resume(
                        name=name or parsed_data.get('name', ''),
                        email=email or parsed_data.get('email', ''),
                        phone=phone or parsed_data.get('phone', ''),
                        resume_text=parsed_data.get('raw_text', ''),
                        skills=final_skills,
                        education=parsed_data.get('education', ''),
                        experience=parsed_data.get('experience', ''),
                        filename=uploaded_file.name
                    )
                    
                    st.success(f"✅ Resume saved successfully! (ID: {resume_id})")
                    st.balloons()
            else:
                st.error("❌ Failed to parse resume. Please try a different file.")
        
        finally:
            # Clean up temporary file
            if os.path.exists(tmp_file_path):
                os.unlink(tmp_file_path)

elif page == "🔍 Evaluate Resumes":
    st.header("Evaluate Resumes Against Job Requirements")
    
    # Get available jobs
    jobs_df = db.get_job_descriptions()
    resumes_df = db.get_resumes()
    
    if jobs_df.empty:
        st.warning("⚠️ No job descriptions found. Please upload a job description first.")
    elif resumes_df.empty:
        st.warning("⚠️ No resumes found. Please upload some resumes first.")
    else:
        # Job selection
        job_options = {f"{row['title']} - {row['company']} (ID: {row['id']})": row['id'] 
                      for _, row in jobs_df.iterrows()}
        
        selected_job_display = st.selectbox("Select Job Description", list(job_options.keys()))
        selected_job_id = job_options[selected_job_display]
        
        # Get job details
        job_row = jobs_df[jobs_df['id'] == selected_job_id].iloc[0]
        job_data = {
            'title': job_row['title'],
            'must_have_skills': json.loads(job_row['must_have_skills']),
            'good_to_have_skills': json.loads(job_row['good_to_have_skills']),
            'raw_text': job_row['description']
        }
        
        # Display job details
        with st.expander("Job Details"):
            st.write(f"**Title:** {job_data['title']}")
            st.write(f"**Must-have Skills:** {', '.join(job_data['must_have_skills'])}")
            st.write(f"**Good-to-have Skills:** {', '.join(job_data['good_to_have_skills'])}")
        
        if st.button("🚀 Evaluate All Resumes"):
            progress_bar = st.progress(0)
            results = []
            
            for idx, (_, resume_row) in enumerate(resumes_df.iterrows()):
                # Prepare resume data
                resume_data = {
                    'name': resume_row['name'],
                    'email': resume_row['email'],
                    'skills': json.loads(resume_row['skills']) if resume_row['skills'] else [],
                    'raw_text': resume_row['resume_text']
                }
                
                # Calculate scores
                scores = matching_engine.calculate_overall_score(resume_data, job_data)
                
                # Generate feedback
                feedback = matching_engine.generate_feedback(resume_data, job_data, scores)
                
                # Save evaluation
                db.add_evaluation(
                    job_id=selected_job_id,
                    resume_id=resume_row['id'],
                    relevance_score=scores['overall_score'],
                    verdict=scores['verdict'],
                    missing_skills=scores['missing_skills'],
                    feedback=feedback,
                    hard_match_score=scores['hard_match_score'],
                    semantic_match_score=scores['semantic_match_score']
                )
                
                results.append({
                    'Name': resume_data['name'],
                    'Email': resume_data['email'],
                    'Overall Score': scores['overall_score'],
                    'Hard Match': scores['hard_match_score'],
                    'Semantic Match': scores['semantic_match_score'],
                    'Verdict': scores['verdict'],
                    'Feedback': feedback
                })
                
                progress_bar.progress((idx + 1) / len(resumes_df))
            
            st.success("✅ Evaluation completed!")
            
            # Display results
            results_df = pd.DataFrame(results)
            
            # Summary metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                high_count = len(results_df[results_df['Verdict'] == 'High'])
                st.metric("High Matches", high_count)
            with col2:
                medium_count = len(results_df[results_df['Verdict'] == 'Medium'])
                st.metric("Medium Matches", medium_count)
            with col3:
                avg_score = results_df['Overall Score'].mean()
                st.metric("Average Score", f"{avg_score:.1f}")
            
            # Results table
            st.subheader("Evaluation Results")
            
            # Color-code verdicts
            def color_verdict(val):
                if val == 'High':
                    return 'background-color: #d4edda'
                elif val == 'Medium':
                    return 'background-color: #fff3cd'
                else:
                    return 'background-color: #f8d7da'
            
            styled_df = results_df.style.applymap(color_verdict, subset=['Verdict'])
            st.dataframe(styled_df, use_container_width=True)
            
            # Download results
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Results as CSV",
                data=csv,
                file_name=f"evaluation_results_{selected_job_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )

elif page == "📈 Analytics":
    st.header("System Analytics")
    
    # Get evaluation data
    evaluations_query = """
        SELECT e.*, j.title as job_title, j.company, r.name as candidate_name
        FROM evaluations e
        JOIN job_descriptions j ON e.job_id = j.id
        JOIN resumes r ON e.resume_id = r.id
        ORDER BY e.created_at DESC
    """
    
    try:
        with db.db_path as conn_path:
            conn = db.db_path
        evaluations_df = pd.read_sql_query(evaluations_query, f"sqlite:///{conn}")
        
        if not evaluations_df.empty:
            # Score distribution
            st.subheader("Score Distribution")
            fig_hist = px.histogram(evaluations_df, x='relevance_score', nbins=20, 
                                   title="Distribution of Relevance Scores")
            st.plotly_chart(fig_hist, use_container_width=True)
            
            # Verdict by job
            st.subheader("Verdict Distribution by Job")
            verdict_by_job = evaluations_df.groupby(['job_title', 'verdict']).size().unstack(fill_value=0)
            fig_bar = px.bar(verdict_by_job, title="Verdict Distribution by Job Position")
            st.plotly_chart(fig_bar, use_container_width=True)
            
            # Top candidates
            st.subheader("Top Performing Candidates")
            top_candidates = evaluations_df.nlargest(10, 'relevance_score')[
                ['candidate_name', 'job_title', 'relevance_score', 'verdict']
            ]
            st.dataframe(top_candidates, use_container_width=True)
            
            # Hard vs Semantic match correlation
            st.subheader("Hard Match vs Semantic Match Analysis")
            fig_scatter = px.scatter(evaluations_df, x='hard_match_score', y='semantic_match_score',
                                   color='verdict', title="Hard Match vs Semantic Match Scores")
            st.plotly_chart(fig_scatter, use_container_width=True)
            
        else:
            st.info("No evaluation data available yet. Run some evaluations first.")
            
    except Exception as e:
        st.error(f"Error loading analytics data: {e}")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        <p>🏢 Innomatics Research Labs - Automated Resume Relevance Check System</p>
        <p>Built with ❤️ using Streamlit, Python, and AI</p>
    </div>
    """, 
    unsafe_allow_html=True
)
