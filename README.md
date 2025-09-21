AI Resume Relevance Checker
An intelligent, scalable web application designed to automate the initial stages of resume screening. This tool leverages a powerful AI language model to analyze and rank multiple candidates against a single job description based on semantic relevance, not just keyword matching.
Features
* Batch Processing: Upload and analyze dozens of resumes simultaneously against one job description.
* Semantic Similarity Scoring: Uses a Sentence-BERT model to understand the context and meaning of the text, providing a highly accurate relevance score.
* Detailed Skill Analysis: Automatically identifies and categorizes skills into Matched Skills, Potential Gaps, and the candidate's Unique Skills.
* Professional Dashboard: Presents a clean, interactive dashboard with a sortable table of all candidates.
* Visual Ranking: Includes a dynamic line chart to easily visualize the score distribution across all candidates.
* Modern UI: Built with a polished, dark-themed, and user-friendly interface.
How It Works
The application follows a sophisticated pipeline to ensure accurate and fast analysis:
1. PDF Parsing: Extracts raw text content from all uploaded PDF documents (resumes and JD).
2. Text Preprocessing: Cleans the text by converting it to lowercase, removing special characters, and standardizing whitespace. This is crucial for accurate comparison.
3. Semantic Encoding: The cleaned text for the JD and each resume is fed into the all-MiniLM-L6-v2 model, which converts the text into numerical vectors (embeddings) that represent its meaning.
4. Similarity Calculation: The tool calculates the cosine similarity between the job description's vector and each resume's vector. This value is then converted to a 0-100% relevance score.
5. Report Generation: A detailed report is generated for each candidate, including an executive summary and a full skill analysis. The top-scoring candidate's report is highlighted.
6. Data Visualization: All results are compiled and displayed in an interactive chart and a sortable data table.
Tech Stack
* Backend & Frontend: Streamlit
* AI / NLP Model: Sentence-Transformers (SBERT)
* Data Handling: Pandas
* PDF Extraction: pdfplumber
* DOCX Extraction: python-docx
* Text Manipulation: re (Regular Expressions)
Setup and Installation
To run this project locally, please follow these steps:
1. Clone the repository:
git clone [https://github.com/Adhiraj36/ats](https://github.com/Adhiraj36/ats)
cd resume-screener

2. Create a virtual environment (recommended):
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

3. Install the required libraries:
pip install -r requirements.txt

How to Run
Once the installation is complete, you can run the application with a single command:
streamlit run app.py

The application will open in a new tab in your web browser.
Future Improvements
* [ ] Extract structured data like contact information, years of experience, and education.
* [ ] Integrate with an Applicant Tracking System (ATS) API.
* [ ] Allow for customizable skill lists to be added via the UI.
* [ ] Deploy the application to a cloud service for wider access.
* Made by : Shaders.
