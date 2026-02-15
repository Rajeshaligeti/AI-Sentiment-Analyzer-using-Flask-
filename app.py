"""
E-Commerce Review Intelligence Platform
Upgraded Flask Sentiment Analyzer for product review analysis
"""

from flask import Flask, render_template, request, jsonify, send_from_directory
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import os
import re
import chardet
from dotenv import load_dotenv
from werkzeug.utils import secure_filename
import json
from datetime import datetime
import logging

# Import custom modules
from modules.scraper import ReviewScraper
from modules.sentiment_analyzer import EnhancedSentimentAnalyzer, AspectSentimentAnalyzer
from modules.llm_engine import ReviewAnalysisEngine, InsightAggregator
from config import get_config

# Load environment variables
load_dotenv()

# Load app configuration
config = get_config()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Configuration
app.config['MAX_CONTENT_LENGTH'] = int(os.getenv('MAX_CONTENT_LENGTH', 16777216))  # 16MB
app.config['UPLOAD_FOLDER'] = os.getenv('UPLOAD_FOLDER', 'uploads')
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'your-secret-key-here')

# Create upload directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('static', exist_ok=True)

# Allowed file extensions
ALLOWED_EXTENSIONS = {'txt', 'csv', 'xlsx', 'xls'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Initialize AI components
def load_models():
    """Load and initialize AI models"""
    # Gemini API is initialized in ReviewAnalysisEngine
    return {
        'sentiment_analyzer': EnhancedSentimentAnalyzer(),
        'review_engine': ReviewAnalysisEngine(
            prefer_local=config.PREFER_LOCAL_LLM,
            api_key=config.GOOGLE_API_KEY
        ),
        'scraper': ReviewScraper()
    }

models = load_models()

# ==================== UTILITY FUNCTIONS ====================

def generate_wordcloud(text):
    """Generate and save wordcloud"""
    try:
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color="white",
            colormap="viridis",
            max_words=100
        ).generate(text)
        
        wordcloud_path = os.path.join('static', 'wordcloud.png')
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.tight_layout(pad=0)
        plt.savefig(wordcloud_path, format='png', dpi=100, bbox_inches='tight')
        plt.close()
        return wordcloud_path
    except Exception as e:
        logger.error(f"Error generating wordcloud: {e}")
        return None

def analyze_batch_reviews(reviews_list):
    """Analyze a batch of reviews"""
    results = []
    sentiment_stats = {'positive': 0, 'negative': 0, 'neutral': 0}
    all_emotions = {}
    all_aspects = {}
    
    for idx, review_text in enumerate(reviews_list[:100], 1):
        if not review_text or len(review_text.strip()) < 5:
            continue
        
        # Sentiment analysis
        sentiment_result = models['sentiment_analyzer'].analyze(review_text)
        
        # Update statistics
        sentiment = sentiment_result['sentiment']
        sentiment_stats[sentiment] += 1
        
        # Aggregate emotions
        for emotion, conf in sentiment_result.get('emotions', {}).items():
            all_emotions[emotion] = all_emotions.get(emotion, 0) + conf
        
        # Aggregate aspects
        for aspect, data in sentiment_result.get('aspects', {}).items():
            if aspect not in all_aspects:
                all_aspects[aspect] = {'positive': 0, 'negative': 0, 'neutral': 0}
            all_aspects[aspect][data['sentiment']] += 1
        
        # Prepare result entry
        results.append({
            'id': idx,
            'text': review_text[:150] + "..." if len(review_text) > 150 else review_text,
            'full_text': review_text,
            'sentiment': sentiment,
            'confidence': float(sentiment_result['confidence']),
            'polarity': float(sentiment_result['polarity']),
            'emotions': sentiment_result.get('emotions', {}),
            'dominant_emotion': sentiment_result.get('dominant_emotion'),
            'aspects': sentiment_result.get('aspects', {})
        })
    
    # Normalize emotion averages
    if results:
        for emotion in all_emotions:
            all_emotions[emotion] = all_emotions[emotion] / len(results)
    
    return results, sentiment_stats, all_emotions, all_aspects

# ==================== ROUTES ====================

@app.route("/", methods=["GET"])
def home():
    """Home page - dashboard"""
    return render_template("dashboard.html")

# ==================== TEXT ANALYSIS ROUTES ====================

@app.route("/api/analyze-text", methods=["POST"])
def analyze_text():
    """Analyze single text input"""
    try:
        data = request.get_json()
        text = data.get('text', '').strip()
        
        if not text or len(text) < 5:
            return jsonify({"error": "Text too short"}), 400
        
        # Analyze sentiment
        result = models['sentiment_analyzer'].analyze(text)
        
        # Generate wordcloud
        wordcloud_path = generate_wordcloud(text)
        
        return jsonify({
            'success': True,
            'analysis': result,
            'wordcloud': wordcloud_path is not None
        })
    
    except Exception as e:
        logger.error(f"Error in text analysis: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/batch-analyze", methods=["POST"])
def batch_analyze():
    """Batch analysis of multiple texts"""
    try:
        data = request.get_json()
        reviews = data.get('reviews', [])
        
        if not reviews or len(reviews) == 0:
            return jsonify({"error": "No reviews provided"}), 400
        
        # Analyze all reviews
        results, sentiment_stats, emotions, aspects = analyze_batch_reviews(reviews)
        
        # Generate wordcloud from all texts
        combined_text = " ".join(reviews)
        wordcloud_path = generate_wordcloud(combined_text)
        
        return jsonify({
            'success': True,
            'total_processed': len(results),
            'results': results,
            'statistics': {
                'sentiment_distribution': sentiment_stats,
                'emotions': emotions,
                'aspects': aspects
            },
            'wordcloud': wordcloud_path is not None
        })
    
    except Exception as e:
        logger.error(f"Error in batch analysis: {e}")
        return jsonify({"error": str(e)}), 500

# ==================== FILE UPLOAD ROUTES ====================

@app.route("/api/upload-file", methods=["POST"])
def upload_file():
    """Upload and analyze file"""
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file part"}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        if not allowed_file(file.filename):
            return jsonify({"error": "File type not allowed. Use .csv, .txt, .xlsx, or .xls"}), 400
        
        # Save file temporarily
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Extract texts based on file type
        texts = []
        try:
            if filename.endswith('.csv'):
                df = pd.read_csv(filepath)
                texts = df.iloc[:, 0].astype(str).tolist()
            elif filename.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(filepath)
                texts = df.iloc[:, 0].astype(str).tolist()
            else:  # .txt
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                    texts = [line.strip() for line in f.readlines() if line.strip()]
        finally:
            # Clean up temp file
            if os.path.exists(filepath):
                os.remove(filepath)
        
        # Analyze all texts
        results, sentiment_stats, emotions, aspects = analyze_batch_reviews(texts)
        
        # Generate wordcloud
        combined_text = " ".join(texts)
        wordcloud_path = generate_wordcloud(combined_text)
        
        return jsonify({
            'success': True,
            'total_processed': len(results),
            'results': results,
            'statistics': {
                'sentiment_distribution': sentiment_stats,
                'emotions': emotions,
                'aspects': aspects
            },
            'wordcloud': wordcloud_path is not None
        })
    
    except Exception as e:
        logger.error(f"Error processing file: {e}")
        return jsonify({"error": str(e)}), 500

# ==================== WEB SCRAPING ROUTES ====================

@app.route("/api/scrape-reviews", methods=["POST"])
def scrape_reviews():
    """Scrape reviews from product URL"""
    try:
        data = request.get_json()
        url = data.get('url', '').strip()
        
        if not url:
            return jsonify({"error": "URL required"}), 400
        
        # Add protocol if missing
        if not url.startswith('http'):
            url = 'https://' + url
        
        logger.info(f"Attempting to scrape: {url}")
        
        # Scrape reviews
        reviews = models['scraper'].scrape_reviews(url, max_reviews=50)
        
        if not reviews:
            return jsonify({
                "error": "Could not scrape reviews from this URL",
                "suggestions": [
                    "✓ Ensure the URL is a product page (not search results)",
                    "✓ Try a different product URL",
                    "✓ Check that reviews are visible in your browser",
                    "✓ Some sites block automated scraping - try the 'Text Input' tab instead",
                    "✓ Supported: Amazon product pages, Flipkart product pages, generic review pages"
                ],
                "debug_url": url
            }), 400
        
        # Extract review texts
        review_texts = [r['text'] for r in reviews]
        ratings = [r.get('rating') for r in reviews if r.get('rating')]
        
        # Analyze reviews
        results, sentiment_stats, emotions, aspects = analyze_batch_reviews(review_texts)
        
        # Generate wordcloud
        combined_text = " ".join(review_texts)
        wordcloud_path = generate_wordcloud(combined_text)
        
        logger.info(f"Successfully analyzed {len(results)} reviews from {url}")
        
        return jsonify({
            'success': True,
            'source': url,
            'total_scraped': len(reviews),
            'total_analyzed': len(results),
            'results': results,
            'ratings': {
                'average': sum(ratings) / len(ratings) if ratings else None,
                'count': len(ratings)
            },
            'statistics': {
                'sentiment_distribution': sentiment_stats,
                'emotions': emotions,
                'aspects': aspects
            },
            'wordcloud': wordcloud_path is not None
        })
    
    except Exception as e:
        logger.error(f"Error scraping reviews: {e}")
        return jsonify({"error": str(e)}), 500

# ==================== LLM INSIGHT ROUTES ====================

@app.route("/api/generate-insights", methods=["POST"])
def generate_insights():
    """Generate business insights using LLM"""
    try:
        data = request.get_json()
        reviews = data.get('reviews', [])
        ratings = data.get('ratings', [])
        
        if not reviews or len(reviews) < 3:
            return jsonify({"error": "At least 3 reviews required"}), 400
        
        # Generate insights
        summary = models['review_engine'].summarize_reviews(reviews)
        business_insights = models['review_engine'].generate_business_insights(reviews, ratings)
        aspects = models['review_engine'].extract_aspects(reviews)
        
        # Identify main complaints and strengths
        results, _, _, _ = analyze_batch_reviews(reviews)
        
        negative_reviews = [r['full_text'] for r in results if r['sentiment'] == 'negative'][:5]
        positive_reviews = [r['full_text'] for r in results if r['sentiment'] == 'positive'][:5]
        
        recommendations = models['review_engine'].generate_improvement_recommendations(negative_reviews) if negative_reviews else None
        marketing_angle = models['review_engine'].generate_marketing_angle([r['full_text'] for r in results if r['sentiment'] == 'positive'][:3]) if positive_reviews else None
        
        return jsonify({
            'success': True,
            'summary': summary,
            'business_insights': business_insights,
            'aspects': aspects,
            'recommendations': recommendations,
            'marketing_angle': marketing_angle,
            'timestamp': datetime.now().isoformat()
        })
    
    except Exception as e:
        logger.error(f"Error generating insights: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/aspect-analysis", methods=["POST"])
def aspect_analysis():
    """Detailed aspect sentiment analysis"""
    try:
        data = request.get_json()
        reviews = data.get('reviews', [])
        
        if not reviews:
            return jsonify({"error": "Reviews required"}), 400
        
        # Analyze each review for aspects
        all_aspects = {}
        
        for review in reviews[:50]:
            aspects = AspectSentimentAnalyzer.analyze_all_aspects(review)
            
            for aspect, data in aspects.items():
                if aspect not in all_aspects:
                    all_aspects[aspect] = {'positive': 0, 'negative': 0, 'neutral': 0, 'total': 0, 'sentences': []}
                
                sentiment = data['sentiment']
                all_aspects[aspect][sentiment] += 1
                all_aspects[aspect]['total'] += 1
                all_aspects[aspect]['sentences'].extend(data.get('sentences', [])[:1])
        
        # Calculate percentages
        aspect_summary = {}
        for aspect, stats in all_aspects.items():
            if stats['total'] > 0:
                aspect_summary[aspect] = {
                    'total_mentions': stats['total'],
                    'positive_pct': (stats['positive'] / stats['total']) * 100,
                    'negative_pct': (stats['negative'] / stats['total']) * 100,
                    'neutral_pct': (stats['neutral'] / stats['total']) * 100,
                    'key_phrases': list(set(stats['sentences'][:3]))
                }
        
        return jsonify({
            'success': True,
            'aspects': aspect_summary
        })
    
    except Exception as e:
        logger.error(f"Error in aspect analysis: {e}")
        return jsonify({"error": str(e)}), 500

# ==================== STATIC FILES ROUTES ====================

@app.route('/download_wordcloud')
def download_wordcloud():
    """Download generated wordcloud"""
    try:
        return send_from_directory('static', 'wordcloud.png', as_attachment=True)
    except Exception as e:
        logger.error(f"Error downloading wordcloud: {e}")
        return jsonify({"error": "Wordcloud not found"}), 404

@app.route('/static/<path:filename>')
def serve_static(filename):
    """Serve static files"""
    return send_from_directory('static', filename)

# ==================== ERROR HANDLERS ====================

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Not found"}), 404

@app.errorhandler(500)
def server_error(error):
    return jsonify({"error": "Server error"}), 500

if __name__ == "__main__":
    # Create __init__.py for modules package if it doesn't exist
    os.makedirs('modules', exist_ok=True)
    if not os.path.exists('modules/__init__.py'):
        open('modules/__init__.py', 'w').close()
    
    # Run development server
    app.run(debug=True, host='0.0.0.0', port=5000)
