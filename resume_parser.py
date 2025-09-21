import fitz  # PyMuPDF
import pdfplumber
import docx2txt
import re
import spacy
from typing import Dict, List, Optional
import logging

class ResumeParser:
    def __init__(self):
        # Load spaCy model for NER
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logging.warning("spaCy model not found. Install with: python -m spacy download en_core_web_sm")
            self.nlp = None
    
    def extract_text_from_pdf(self, file_path: str) -> str:
        """Extract text from PDF using multiple methods for better accuracy"""
        text = ""
        
        # Method 1: PyMuPDF
        try:
            doc = fitz.open(file_path)
            for page in doc:
                text += page.get_text()
            doc.close()
        except:
            # Method 2: pdfplumber as fallback
            try:
                with pdfplumber.open(file_path) as pdf:
                    for page in pdf.pages:
                        text += page.extract_text() or ""
            except Exception as e:
                logging.error(f"Error extracting PDF: {e}")
        
        return self.clean_text(text)
    
    def extract_text_from_docx(self, file_path: str) -> str:
        """Extract text from DOCX file"""
        try:
            text = docx2txt.process(file_path)
            return self.clean_text(text)
        except Exception as e:
            logging.error(f"Error extracting DOCX: {e}")
            return ""
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s@.-]', '', text)
        return text.strip()
    
    def extract_contact_info(self, text: str) -> Dict[str, str]:
        """Extract contact information using regex patterns"""
        contact_info = {"name": "", "email": "", "phone": ""}
        
        # Email extraction
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails = re.findall(email_pattern, text)
        if emails:
            contact_info["email"] = emails[0]
        
        # Phone extraction
        phone_pattern = r'(\+?\d{1,3}[-.\s]?)?\(?(\d{3})\)?[-.\s]?(\d{3})[-.\s]?(\d{4})'
        phones = re.findall(phone_pattern, text)
        if phones:
            contact_info["phone"] = ''.join(phones[0])
        
        # Name extraction (first line or before email)
        lines = text.split('\n')
        for line in lines[:5]:  # Check first 5 lines
            line = line.strip()
            if len(line.split()) >= 2 and len(line.split()) <= 4:
                if not any(char.isdigit() or char in '@.-' for char in line):
                    contact_info["name"] = line
                    break
        
        return contact_info
    
    def extract_skills(self, text: str) -> List[str]:
        """Extract skills from resume text"""
        # Common technical skills database
        skill_patterns = [
            # Programming languages
            r'\b(?:Python|Java|JavaScript|C\+\+|C#|PHP|Ruby|Go|Rust|Swift|Kotlin|Scala|R|MATLAB|SQL|HTML|CSS)\b',
            # Frameworks
            r'\b(?:React|Angular|Vue|Django|Flask|Spring|Node\.js|Express|Laravel|Rails|ASP\.NET)\b',
            # Databases
            r'\b(?:MySQL|PostgreSQL|MongoDB|Oracle|SQLite|Redis|Cassandra|DynamoDB)\b',
            # Cloud platforms
            r'\b(?:AWS|Azure|GCP|Docker|Kubernetes|Jenkins|Git|GitHub|GitLab)\b',
            # Data Science
            r'\b(?:Machine Learning|Deep Learning|TensorFlow|PyTorch|Pandas|NumPy|Scikit-learn|Tableau|Power BI)\b',
            # Other technical skills
            r'\b(?:Agile|Scrum|DevOps|CI/CD|REST|API|Microservices|Linux|Windows|macOS)\b'
        ]
        
        skills = []
        text_upper = text.upper()
        
        for pattern in skill_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            skills.extend(matches)
        
        # Remove duplicates and return
        return list(set([skill.strip() for skill in skills if len(skill.strip()) > 1]))
    
    def extract_education(self, text: str) -> str:
        """Extract education information"""
        education_keywords = ['education', 'qualification', 'degree', 'university', 'college', 'bachelor', 'master', 'phd', 'diploma']
        education_section = ""
        
        lines = text.lower().split('\n')
        education_start = -1
        
        for i, line in enumerate(lines):
            if any(keyword in line for keyword in education_keywords):
                education_start = i
                break
        
        if education_start != -1:
            # Extract next 10 lines after education keyword
            education_lines = lines[education_start:education_start + 10]
            education_section = '\n'.join(education_lines)
        
        return education_section.strip()
    
    def extract_experience(self, text: str) -> str:
        """Extract work experience information"""
        experience_keywords = ['experience', 'employment', 'work', 'career', 'position', 'job']
        experience_section = ""
        
        lines = text.lower().split('\n')
        experience_start = -1
        
        for i, line in enumerate(lines):
            if any(keyword in line for keyword in experience_keywords):
                experience_start = i
                break
        
        if experience_start != -1:
            # Extract next 15 lines after experience keyword
            experience_lines = lines[experience_start:experience_start + 15]
            experience_section = '\n'.join(experience_lines)
        
        return experience_section.strip()
    
    def parse_resume(self, file_path: str, file_type: str) -> Dict:
        """Main method to parse resume and extract all information"""
        try:
            # Extract text based on file type
            if file_type.lower() == 'pdf':
                text = self.extract_text_from_pdf(file_path)
            elif file_type.lower() in ['docx', 'doc']:
                text = self.extract_text_from_docx(file_path)
            else:
                raise ValueError("Unsupported file type")
            
            # Extract various components
            contact_info = self.extract_contact_info(text)
            skills = self.extract_skills(text)
            education = self.extract_education(text)
            experience = self.extract_experience(text)
            
            return {
                'raw_text': text,
                'name': contact_info.get('name', ''),
                'email': contact_info.get('email', ''),
                'phone': contact_info.get('phone', ''),
                'skills': skills,
                'education': education,
                'experience': experience
            }
            
        except Exception as e:
            logging.error(f"Error parsing resume: {e}")
            return None
