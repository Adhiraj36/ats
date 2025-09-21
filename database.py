
import sqlite3
import pandas as pd
from datetime import datetime
import json
from typing import Dict, List, Optional
import logging

class DatabaseManager:
    def __init__(self, db_path: str = "resume_system.db"):
        self.db_path = db_path
        self.init_database()

    def init_database(self):
        """Initialize database with required tables"""
        with sqlite3.connect(self.db_path) as conn:
            # Job descriptions table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS job_descriptions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    title TEXT NOT NULL,
                    company TEXT,
                    description TEXT NOT NULL,
                    must_have_skills TEXT,
                    good_to_have_skills TEXT,
                    location TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # Resumes table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS resumes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT,
                    email TEXT,
                    phone TEXT,
                    resume_text TEXT NOT NULL,
                    skills TEXT,
                    education TEXT,
                    experience TEXT,
                    filename TEXT,
                    uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # Evaluations table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS evaluations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id INTEGER,
                    resume_id INTEGER,
                    relevance_score REAL,
                    verdict TEXT,
                    missing_skills TEXT,
                    feedback TEXT,
                    hard_match_score REAL,
                    semantic_match_score REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (job_id) REFERENCES job_descriptions (id),
                    FOREIGN KEY (resume_id) REFERENCES resumes (id)
                )
            ''')

    def add_job_description(self, title: str, company: str, description: str, 
                           must_have_skills: List[str], good_to_have_skills: List[str], 
                           location: str = "") -> int:
        """Add a job description to database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO job_descriptions 
                (title, company, description, must_have_skills, good_to_have_skills, location)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (title, company, description, 
                 json.dumps(must_have_skills), 
                 json.dumps(good_to_have_skills), location))
            return cursor.lastrowid

    def add_resume(self, name: str, email: str, phone: str, resume_text: str, 
                   skills: List[str], education: str, experience: str, filename: str) -> int:
        """Add a resume to database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO resumes 
                (name, email, phone, resume_text, skills, education, experience, filename)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (name, email, phone, resume_text, 
                 json.dumps(skills), education, experience, filename))
            return cursor.lastrowid

    def add_evaluation(self, job_id: int, resume_id: int, relevance_score: float, 
                      verdict: str, missing_skills: List[str], feedback: str,
                      hard_match_score: float, semantic_match_score: float) -> int:
        """Add an evaluation to database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO evaluations 
                (job_id, resume_id, relevance_score, verdict, missing_skills, 
                 feedback, hard_match_score, semantic_match_score)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (job_id, resume_id, relevance_score, verdict, 
                 json.dumps(missing_skills), feedback, hard_match_score, semantic_match_score))
            return cursor.lastrowid

    def get_job_descriptions(self) -> pd.DataFrame:
        """Get all job descriptions"""
        with sqlite3.connect(self.db_path) as conn:
            return pd.read_sql_query("SELECT * FROM job_descriptions ORDER BY created_at DESC", conn)

    def get_resumes(self) -> pd.DataFrame:
        """Get all resumes"""
        with sqlite3.connect(self.db_path) as conn:
            return pd.read_sql_query("SELECT * FROM resumes ORDER BY uploaded_at DESC", conn)

    def get_evaluations_by_job(self, job_id: int) -> pd.DataFrame:
        """Get evaluations for a specific job with resume details"""
        with sqlite3.connect(self.db_path) as conn:
            query = '''
                SELECT e.*, r.name, r.email, r.phone, r.filename
                FROM evaluations e
                JOIN resumes r ON e.resume_id = r.id
                WHERE e.job_id = ?
                ORDER BY e.relevance_score DESC
            '''
            return pd.read_sql_query(query, conn, params=(job_id,))

    def get_evaluation_stats(self) -> Dict:
        """Get evaluation statistics"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Total counts
            cursor.execute("SELECT COUNT(*) FROM job_descriptions")
            total_jobs = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM resumes")
            total_resumes = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM evaluations")
            total_evaluations = cursor.fetchone()[0]

            # Verdict distribution
            cursor.execute("SELECT verdict, COUNT(*) FROM evaluations GROUP BY verdict")
            verdict_dist = dict(cursor.fetchall())

            return {
                'total_jobs': total_jobs,
                'total_resumes': total_resumes,
                'total_evaluations': total_evaluations,
                'verdict_distribution': verdict_dist
            }
