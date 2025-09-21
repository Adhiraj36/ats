import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import fuzz
import logging
from typing import Dict, List, Tuple

class MatchingEngine:
    def __init__(self):
        # Initialize semantic similarity model
        try:
            self.semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
            logging.info("Semantic model loaded successfully")
        except Exception as e:
            logging.error(f"Error loading semantic model: {e}")
            self.semantic_model = None
        
        # Initialize TF-IDF vectorizer for keyword matching
        self.tfidf = TfidfVectorizer(stop_words='english', max_features=1000)
    
    def calculate_hard_match_score(self, resume_data: Dict, job_data: Dict) -> Tuple[float, List[str]]:
        """Calculate hard match score based on exact and fuzzy keyword matching"""
        resume_skills = set([skill.lower() for skill in resume_data.get('skills', [])])
        must_have_skills = set([skill.lower() for skill in job_data.get('must_have_skills', [])])
        good_to_have_skills = set([skill.lower() for skill in job_data.get('good_to_have_skills', [])])
        
        # Exact matches
        exact_must_have_matches = len(resume_skills.intersection(must_have_skills))
        exact_good_to_have_matches = len(resume_skills.intersection(good_to_have_skills))
        
        # Fuzzy matches
        fuzzy_must_have_matches = 0
        fuzzy_good_to_have_matches = 0
        missing_skills = []
        
        # Check fuzzy matches for must-have skills
        for must_skill in must_have_skills:
            best_match_score = 0
            for resume_skill in resume_skills:
                score = fuzz.ratio(must_skill, resume_skill)
                best_match_score = max(best_match_score, score)
            
            if best_match_score >= 80:  # 80% similarity threshold
                fuzzy_must_have_matches += 1
            else:
                missing_skills.append(must_skill)
        
        # Check fuzzy matches for good-to-have skills
        for good_skill in good_to_have_skills:
            best_match_score = 0
            for resume_skill in resume_skills:
                score = fuzz.ratio(good_skill, resume_skill)
                best_match_score = max(best_match_score, score)
            
            if best_match_score >= 80:
                fuzzy_good_to_have_matches += 1
        
        # Calculate scores
        total_must_have = len(must_have_skills)
        total_good_to_have = len(good_to_have_skills)
        
        if total_must_have == 0 and total_good_to_have == 0:
            return 0.0, missing_skills
        
        must_have_score = (exact_must_have_matches + fuzzy_must_have_matches) / max(total_must_have, 1)
        good_to_have_score = (exact_good_to_have_matches + fuzzy_good_to_have_matches) / max(total_good_to_have, 1)
        
        # Weighted score (must-have skills are more important)
        hard_match_score = (must_have_score * 0.7 + good_to_have_score * 0.3) * 100
        
        return min(hard_match_score, 100.0), missing_skills
    
    def calculate_semantic_match_score(self, resume_text: str, job_text: str) -> float:
        """Calculate semantic similarity using sentence transformers"""
        if not self.semantic_model:
            return 0.0
        
        try:
            # Generate embeddings
            resume_embedding = self.semantic_model.encode([resume_text])
            job_embedding = self.semantic_model.encode([job_text])
            
            # Calculate cosine similarity
            similarity = cosine_similarity(resume_embedding, job_embedding)[0][0]
            
            # Convert to percentage
            return float(similarity * 100)
            
        except Exception as e:
            logging.error(f"Error calculating semantic similarity: {e}")
            return 0.0
    
    def calculate_tfidf_similarity(self, resume_text: str, job_text: str) -> float:
        """Calculate TF-IDF based similarity as fallback"""
        try:
            # Fit TF-IDF on both texts
            documents = [resume_text, job_text]
            tfidf_matrix = self.tfidf.fit_transform(documents)
            
            # Calculate cosine similarity
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            
            return float(similarity * 100)
            
        except Exception as e:
            logging.error(f"Error calculating TF-IDF similarity: {e}")
            return 0.0
    
    def calculate_overall_score(self, resume_data: Dict, job_data: Dict) -> Dict:
        """Calculate overall relevance score combining hard and semantic matching"""
        
        # Calculate hard match score
        hard_score, missing_skills = self.calculate_hard_match_score(resume_data, job_data)
        
        # Calculate semantic match score
        resume_text = resume_data.get('raw_text', '')
        job_text = job_data.get('raw_text', '')
        
        semantic_score = self.calculate_semantic_match_score(resume_text, job_text)
        
        # If semantic model fails, use TF-IDF as fallback
        if semantic_score == 0.0:
            semantic_score = self.calculate_tfidf_similarity(resume_text, job_text)
        
        # Combine scores with weights
        # Hard matching: 60%, Semantic matching: 40%
        overall_score = (hard_score * 0.6) + (semantic_score * 0.4)
        
        # Determine verdict based on score
        if overall_score >= 75:
            verdict = "High"
        elif overall_score >= 50:
            verdict = "Medium"
        else:
            verdict = "Low"
        
        return {
            'overall_score': round(overall_score, 2),
            'hard_match_score': round(hard_score, 2),
            'semantic_match_score': round(semantic_score, 2),
            'verdict': verdict,
            'missing_skills': missing_skills
        }
    
    def generate_feedback(self, resume_data: Dict, job_data: Dict, scores: Dict) -> str:
        """Generate personalized feedback for the candidate"""
        feedback = []
        
        # Overall assessment
        verdict = scores['verdict']
        overall_score = scores['overall_score']
        
        if verdict == "High":
            feedback.append(f"Excellent match! Your resume shows {overall_score:.1f}% relevance to this position.")
        elif verdict == "Medium":
            feedback.append(f"Good potential match with {overall_score:.1f}% relevance. Some improvements recommended.")
        else:
            feedback.append(f"Limited match with {overall_score:.1f}% relevance. Significant improvements needed.")
        
        # Missing skills feedback
        missing_skills = scores.get('missing_skills', [])
        if missing_skills:
            feedback.append(f"\nSkills to develop: {', '.join(missing_skills[:5])}")
        
        # Specific recommendations
        hard_score = scores['hard_match_score']
        semantic_score = scores['semantic_match_score']
        
        if hard_score < 50:
            feedback.append("Focus on acquiring the technical skills mentioned in the job requirements.")
        
        if semantic_score < 50:
            feedback.append("Consider restructuring your resume to better highlight relevant experience and projects.")
        
        # Positive reinforcement
        resume_skills = resume_data.get('skills', [])
        job_must_have = job_data.get('must_have_skills', [])
        matching_skills = set([s.lower() for s in resume_skills]).intersection(set([s.lower() for s in job_must_have]))
        
        if matching_skills:
            feedback.append(f"\nStrengths: You have relevant experience in {', '.join(list(matching_skills)[:3])}")
        
        return "\n".join(feedback)
