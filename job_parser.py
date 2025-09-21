import re
import json
from typing import Dict, List
import spacy

class JobDescriptionParser:
    def __init__(self):
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            self.nlp = None
    
    def parse_job_description(self, jd_text: str) -> Dict:
        """Parse job description and extract structured information"""
        
        # Extract job title
        title = self.extract_title(jd_text)
        
        # Extract skills
        must_have_skills = self.extract_must_have_skills(jd_text)
        good_to_have_skills = self.extract_good_to_have_skills(jd_text)
        
        # Extract qualifications
        qualifications = self.extract_qualifications(jd_text)
        
        # Extract experience requirements
        experience_req = self.extract_experience_requirements(jd_text)
        
        return {
            'title': title,
            'must_have_skills': must_have_skills,
            'good_to_have_skills': good_to_have_skills,
            'qualifications': qualifications,
            'experience_requirements': experience_req,
            'raw_text': jd_text
        }
    
    def extract_title(self, text: str) -> str:
        """Extract job title from JD"""
        lines = text.split('\n')
        for line in lines[:5]:
            line = line.strip()
            if 'position' in line.lower() or 'role' in line.lower() or 'job title' in line.lower():
                return line
            # First substantial line might be title
            if len(line) > 10 and len(line) < 100:
                return line
        return "Not specified"
    
    def extract_must_have_skills(self, text: str) -> List[str]:
        """Extract must-have skills"""
        must_have_patterns = [
            r'(?i)(?:required|must have|mandatory|essential).*?skills?.*?[:](.*?)(?:\n\n|required|preferred)',
            r'(?i)(?:requirements|required skills?).*?[:](.*?)(?:\n\n|preferred|nice)',
            r'(?i)(?:essential skills?).*?[:](.*?)(?:\n\n|preferred|nice)'
        ]
        
        skills = []
        for pattern in must_have_patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            for match in matches:
                extracted_skills = self.extract_skills_from_text(match)
                skills.extend(extracted_skills)
        
        return list(set(skills))
    
    def extract_good_to_have_skills(self, text: str) -> List[str]:
        """Extract good-to-have skills"""
        good_to_have_patterns = [
            r'(?i)(?:preferred|nice to have|good to have|plus|bonus).*?skills?.*?[:](.*?)(?:\n\n|requirements)',
            r'(?i)(?:additional|desired).*?skills?.*?[:](.*?)(?:\n\n|requirements)',
        ]
        
        skills = []
        for pattern in good_to_have_patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            for match in matches:
                extracted_skills = self.extract_skills_from_text(match)
                skills.extend(extracted_skills)
        
        return list(set(skills))
    
    def extract_skills_from_text(self, text: str) -> List[str]:
        """Extract individual skills from text block"""
        # Common technical skills
        skill_patterns = [
            r'\b(?:Python|Java|JavaScript|C\+\+|C#|PHP|Ruby|Go|Rust|Swift|Kotlin|Scala|R|MATLAB|SQL|HTML|CSS)\b',
            r'\b(?:React|Angular|Vue|Django|Flask|Spring|Node\.js|Express|Laravel|Rails|ASP\.NET)\b',
            r'\b(?:MySQL|PostgreSQL|MongoDB|Oracle|SQLite|Redis|Cassandra|DynamoDB)\b',
            r'\b(?:AWS|Azure|GCP|Docker|Kubernetes|Jenkins|Git|GitHub|GitLab)\b',
            r'\b(?:Machine Learning|Deep Learning|TensorFlow|PyTorch|Pandas|NumPy|Scikit-learn|Tableau|Power BI)\b',
            r'\b(?:Agile|Scrum|DevOps|CI/CD|REST|API|Microservices|Linux|Windows|macOS)\b'
        ]
        
        skills = []
        for pattern in skill_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            skills.extend(matches)
        
        # Also split by common separators
        text_skills = re.split(r'[,;•\n\-\*]', text)
        for skill in text_skills:
            skill = skill.strip()
            if 2 < len(skill) < 30 and not any(char.isdigit() for char in skill):
                skills.append(skill)
        
        return [skill.strip() for skill in skills if len(skill.strip()) > 2]
    
    def extract_qualifications(self, text: str) -> List[str]:
        """Extract educational qualifications"""
        qual_pattern = r'(?i)(?:education|qualification|degree).*?[:](.*?)(?:\n\n|experience|requirements)'
        matches = re.findall(qual_pattern, text, re.DOTALL)
        
        qualifications = []
        for match in matches:
            quals = re.split(r'[,;\n\-\*]', match)
            for qual in quals:
                qual = qual.strip()
                if len(qual) > 5:
                    qualifications.append(qual)
        
        return qualifications
    
    def extract_experience_requirements(self, text: str) -> str:
        """Extract experience requirements"""
        exp_patterns = [
            r'(\d+\+?\s*(?:years?|yrs?)\s*(?:of\s*)?(?:experience|exp))',
            r'(?:experience|exp).*?(\d+\+?\s*(?:years?|yrs?))',
            r'(\d+\-\d+\s*(?:years?|yrs?)\s*(?:of\s*)?(?:experience|exp))'
        ]
        
        for pattern in exp_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                return matches[0]
        
        return "Not specified"
