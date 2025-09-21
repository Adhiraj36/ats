# Configuration file for Resume Relevance Check System

import json

# Scoring Configuration
SCORING_CONFIG = {
    "default_weights": {
        "hard_match": 0.4,
        "semantic_match": 0.6
    },
    "hard_match_components": {
        "keywords": 0.6,
        "skills": 0.4
    },
    "score_thresholds": {
        "high_suitability": 75,
        "medium_suitability": 50,
        "low_suitability": 0
    }
}

# OpenAI Configuration
OPENAI_CONFIG = {
    "embedding_model": "text-embedding-3-small",
    "max_tokens": 8192,
    "batch_size": 100
}

# Text Processing Configuration
TEXT_PROCESSING_CONFIG = {
    "max_keywords": 50,
    "min_keyword_length": 2,
    "remove_stopwords": True,
    "languages": ["english"]
}

# Skills Database - can be expanded based on industry requirements
SKILLS_DATABASE = {
    "programming_languages": [
        "python", "java", "javascript", "typescript", "c++", "c#", "php", "ruby", 
        "go", "swift", "kotlin", "scala", "r", "matlab", "rust", "dart"
    ],
    "web_technologies": [
        "html", "css", "react", "angular", "vue", "svelte", "jquery", "bootstrap",
        "node.js", "express", "django", "flask", "spring", "rails", "laravel"
    ],
    "databases": [
        "mysql", "postgresql", "mongodb", "redis", "elasticsearch", "oracle",
        "sql server", "sqlite", "cassandra", "dynamodb", "neo4j"
    ],
    "cloud_platforms": [
        "aws", "azure", "gcp", "google cloud", "kubernetes", "docker", "terraform",
        "ansible", "jenkins", "gitlab", "github actions"
    ],
    "data_science": [
        "machine learning", "deep learning", "data science", "artificial intelligence",
        "pandas", "numpy", "tensorflow", "pytorch", "scikit-learn", "keras",
        "tableau", "power bi", "excel", "spark", "hadoop"
    ],
    "mobile_development": [
        "android", "ios", "react native", "flutter", "xamarin", "ionic"
    ],
    "other_technologies": [
        "git", "linux", "windows", "agile", "scrum", "rest", "api", "microservices",
        "blockchain", "devops", "cybersecurity", "ui/ux", "photoshop", "figma"
    ]
}

# Contact Information Patterns
CONTACT_PATTERNS = {
    "email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
    "phone": r'\+?\d[\d\s\-\(\)]{8,}\d',
    "linkedin": r'linkedin\.com/in/[\w\-]+',
    "github": r'github\.com/[\w\-]+'
}

# UI Configuration
UI_CONFIG = {
    "page_title": "Resume Relevance Check System",
    "page_icon": "📄",
    "layout": "wide",
    "theme": {
        "primary_color": "#FF6B6B",
        "background_color": "#FFFFFF",
        "secondary_background_color": "#F0F2F6",
        "text_color": "#262730"
    }
}

# File Upload Configuration
FILE_CONFIG = {
    "allowed_types": ["pdf", "docx"],
    "max_file_size_mb": 10,
    "max_files_per_batch": 20
}

# Analytics Configuration
ANALYTICS_CONFIG = {
    "score_ranges": ["0-25", "26-50", "51-75", "76-100"],
    "top_skills_limit": 10,
    "chart_colors": ["#FF6B6B", "#4ECDC4", "#45B7D1", "#FFA07A", "#98D8C8"]
}

def save_config_to_file(filename="config.json"):
    """Save configuration to JSON file"""
    config_data = {
        "scoring": SCORING_CONFIG,
        "openai": OPENAI_CONFIG,
        "text_processing": TEXT_PROCESSING_CONFIG,
        "skills_database": SKILLS_DATABASE,
        "contact_patterns": CONTACT_PATTERNS,
        "ui": UI_CONFIG,
        "file_upload": FILE_CONFIG,
        "analytics": ANALYTICS_CONFIG
    }

    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(config_data, f, indent=2, ensure_ascii=False)

    return config_data

def load_config_from_file(filename="config.json"):
    """Load configuration from JSON file"""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        return save_config_to_file(filename)

# Save default configuration
if __name__ == "__main__":
    config_data = save_config_to_file()
    print("Configuration saved to config.json")
