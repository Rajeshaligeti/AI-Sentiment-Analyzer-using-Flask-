"""
Configuration module for the E-Commerce Review Intelligence Platform
Manages all environment variables and settings
"""

import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    """Base configuration"""
    
    # Flask
    DEBUG = os.getenv('FLASK_ENV') != 'production'
    SECRET_KEY = os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production')
    
    # File Upload
    UPLOAD_FOLDER = os.getenv('UPLOAD_FOLDER', 'uploads')
    MAX_CONTENT_LENGTH = int(os.getenv('MAX_CONTENT_LENGTH', 16777216))  # 16MB
    ALLOWED_EXTENSIONS = {'txt', 'csv', 'xlsx', 'xls'}
    
    # API Keys
    GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
    
    # LLM Configuration
    OLLAMA_MODEL = os.getenv('OLLAMA_MODEL', 'llama2')
    OLLAMA_BASE_URL = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
    PREFER_LOCAL_LLM = os.getenv('PREFER_LOCAL_LLM', 'true').lower() == 'true'
    
    # Scraping Configuration
    MAX_REVIEWS_TO_SCRAPE = int(os.getenv('MAX_REVIEWS_TO_SCRAPE', 50))
    SCRAPE_TIMEOUT = int(os.getenv('SCRAPE_TIMEOUT', 10))
    
    # Analysis Configuration
    SENTIMENT_MODEL = 'gemini-1.5-flash'
    BATCH_SIZE = int(os.getenv('BATCH_SIZE', 100))
    
    # Visualization
    WORDCLOUD_WIDTH = int(os.getenv('WORDCLOUD_WIDTH', 800))
    WORDCLOUD_HEIGHT = int(os.getenv('WORDCLOUD_HEIGHT', 400))
    MAX_WORDCLOUD_WORDS = int(os.getenv('MAX_WORDCLOUD_WORDS', 100))
    
    # Logging
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    LOG_FILE = os.getenv('LOG_FILE', 'app.log')


class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    TESTING = False


class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    TESTING = False
    PREFER_LOCAL_LLM = False  # Use cloud LLM in production


class TestingConfig(Config):
    """Testing configuration"""
    DEBUG = True
    TESTING = True
    MAX_CONTENT_LENGTH = 1048576  # 1MB for testing
    MAX_REVIEWS_TO_SCRAPE = 5


def get_config():
    """Get configuration based on environment"""
    env = os.getenv('FLASK_ENV', 'development')
    
    if env == 'production':
        return ProductionConfig()
    elif env == 'testing':
        return TestingConfig()
    else:
        return DevelopmentConfig()
