import subprocess
import sys
import os

def install_requirements():
    """Install required packages"""
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    
    # Download spaCy model
    subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])

def create_directories():
    """Create necessary directories"""
    directories = ['uploads', 'data', 'logs']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

if __name__ == "__main__":
    print("Setting up Innomatics Resume Relevance Check System...")
    install_requirements()
    create_directories()
    print("Setup completed! Run 'streamlit run app.py' to start the application.")
