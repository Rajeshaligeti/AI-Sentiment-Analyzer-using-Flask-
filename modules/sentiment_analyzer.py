"""
Enhanced sentiment analysis with emotion detection and aspect-based analysis
"""

import re
from typing import Dict, List, Tuple, Optional
from textblob import TextBlob
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AspectSentimentAnalyzer:
    """Analyzes sentiment for specific aspects (price, quality, delivery, support)"""
    
    # Keywords for different aspects
    ASPECT_KEYWORDS = {
        'price': ['price', 'cost', 'expensive', 'cheap', 'affordable', 'worth', 'value', 'rupees', 'dollar', '$', '₹', 'money'],
        'quality': ['quality', 'durable', 'material', 'build', 'craftsmanship', 'defective', 'broken', 'excellent', 'poor'],
        'delivery': ['delivery', 'shipping', 'arrived', 'late', 'damaged', 'packaging', 'fast', 'quick', 'slow', 'arrived'],
        'support': ['support', 'customer', 'service', 'help', 'refund', 'return', 'replacement', 'warranty', 'responsive', 'helpful'],
        'packaging': ['packaging', 'package', 'box', 'wrapped', 'protection', 'damaged', 'unboxing'],
    }
    
    # Sentiment keywords
    POSITIVE_WORDS = {
        'excellent', 'amazing', 'awesome', 'great', 'good', 'love', 'perfect', 'fantastic', 'wonderful',
        'superb', 'best', 'happy', 'satisfied', 'impressed', 'quality', 'durable', 'fast', 'quick',
        'reliable', 'recommend', 'worth', 'value', 'helpful', 'responsive'
    }
    
    NEGATIVE_WORDS = {
        'terrible', 'horrible', 'awful', 'bad', 'hate', 'waste', 'disappointing', 'poor', 'worst',
        'useless', 'broken', 'defective', 'delayed', 'damaged', 'overpriced', 'cheap', 'cheap-quality',
        'unhappy', 'dissatisfied', 'frustrated', 'angry', 'refund', 'return', 'complain', 'issue',
        'problem', 'not recommend'
    }
    
    @staticmethod
    def extract_aspects(text: str) -> Dict[str, List[str]]:
        """Extract aspect mentions from text"""
        text_lower = text.lower()
        aspects_found = {}
        
        for aspect, keywords in AspectSentimentAnalyzer.ASPECT_KEYWORDS.items():
            found = [kw for kw in keywords if kw in text_lower]
            if found:
                aspects_found[aspect] = found
        
        return aspects_found
    
    @staticmethod
    def analyze_aspect_sentiment(text: str, aspect: str) -> Dict:
        """Analyze sentiment for a specific aspect"""
        text_lower = text.lower()
        
        # Find sentences containing aspect keywords
        sentences = re.split(r'[.!?]', text)
        aspect_sentences = []
        
        for sentence in sentences:
            if any(kw in sentence.lower() for kw in AspectSentimentAnalyzer.ASPECT_KEYWORDS.get(aspect, [])):
                aspect_sentences.append(sentence.strip())
        
        if not aspect_sentences:
            return {'aspect': aspect, 'sentiment': 'neutral', 'confidence': 0.0, 'sentences': []}
        
        # Analyze sentiment of aspect sentences
        positive_count = 0
        negative_count = 0
        
        for sentence in aspect_sentences:
            words = set(sentence.lower().split())
            pos_matches = len(words & AspectSentimentAnalyzer.POSITIVE_WORDS)
            neg_matches = len(words & AspectSentimentAnalyzer.NEGATIVE_WORDS)
            
            if pos_matches > neg_matches:
                positive_count += 1
            elif neg_matches > pos_matches:
                negative_count += 1
        
        total = positive_count + negative_count
        if total == 0:
            sentiment = 'neutral'
            confidence = 0.0
        elif positive_count > negative_count:
            sentiment = 'positive'
            confidence = positive_count / total
        else:
            sentiment = 'negative'
            confidence = negative_count / total
        
        return {
            'aspect': aspect,
            'sentiment': sentiment,
            'confidence': confidence,
            'sentences': aspect_sentences[:3]  # Return top 3 relevant sentences
        }
    
    @staticmethod
    def analyze_all_aspects(text: str) -> Dict[str, Dict]:
        """Analyze sentiment for all aspects"""
        aspects = AspectSentimentAnalyzer.extract_aspects(text)
        results = {}
        
        for aspect in aspects.keys():
            results[aspect] = AspectSentimentAnalyzer.analyze_aspect_sentiment(text, aspect)
        
        return results


class EmotionDetector:
    """Detects basic emotions from text"""
    
    EMOTION_KEYWORDS = {
        'joy': ['love', 'happy', 'great', 'awesome', 'wonderful', 'excellent', 'amazing', 'perfect', 'great', '😊', ':)'],
        'sadness': ['sad', 'unhappy', 'terrible', 'awful', 'horrible', 'disappointing', 'worst', 'waste', '😢', ':('],
        'anger': ['angry', 'furious', 'hate', 'frustrated', 'outrageous', 'unacceptable', 'disgusted', '😠'],
        'surprise': ['shocked', 'surprised', 'unexpected', 'amazing', 'wow', 'incredible', '😲'],
        'trust': ['reliable', 'trust', 'safe', 'secure', 'confident', 'good', 'quality'],
        'fear': ['worried', 'concerned', 'scared', 'risky', 'dangerous', 'afraid'],
    }
    
    @staticmethod
    def detect_emotions(text: str) -> Dict[str, float]:
        """Detect emotions in text with confidence scores"""
        text_lower = text.lower()
        emotion_scores = {emotion: 0 for emotion in EmotionDetector.EMOTION_KEYWORDS}
        
        total_matches = 0
        for emotion, keywords in EmotionDetector.EMOTION_KEYWORDS.items():
            matches = sum(1 for kw in keywords if kw in text_lower)
            emotion_scores[emotion] = matches
            total_matches += matches
        
        # Normalize to confidence scores
        if total_matches > 0:
            emotion_scores = {e: score / total_matches for e, score in emotion_scores.items()}
        
        # Return non-zero emotions sorted by confidence
        return {e: score for e, score in sorted(emotion_scores.items(), key=lambda x: x[1], reverse=True) if score > 0}
    
    @staticmethod
    def get_dominant_emotion(emotions: Dict[str, float]) -> Optional[Tuple[str, float]]:
        """Get the dominant emotion"""
        if not emotions:
            return None
        return max(emotions.items(), key=lambda x: x[1])


class EnhancedSentimentAnalyzer:
    """Main enhanced sentiment analyzer"""
    
    # Keyword-based sentiment detection for short texts or when TextBlob is unreliable
    STRONG_POSITIVE = {
        'excellent', 'amazing', 'awesome', 'fantastic', 'wonderful', 'perfect', 'love', 'great',
        'best', 'outstanding', 'superb', 'brilliant', 'incredible', 'exceptional'
    }
    
    STRONG_NEGATIVE = {
        'terrible', 'horrible', 'awful', 'disgusting', 'hate', 'worst', 'poor', 'bad', 'useless',
        'broken', 'defective', 'disappointing', 'pathetic', 'dreadful', 'atrocious'
    }
    
    WEAK_POSITIVE = {
        'good', 'nice', 'ok', 'okay', 'fine', 'decent', 'alright', 'satisfied', 'happy', 'pleased'
    }
    
    WEAK_NEGATIVE = {
        'not good', 'not great', 'disappointed', 'unhappy', 'unsatisfied', 'issue', 'problem',
        'complaint', 'concern', 'mediocre', 'average'
    }
    
    @staticmethod
    def analyze(text: str) -> Dict:
        """
        Comprehensive sentiment analysis
        Returns: sentiment, confidence, emotions, aspect sentiments, polarity
        """
        if not text or not text.strip():
            return {
                'sentiment': 'neutral',
                'confidence': 0.0,
                'polarity': 0.0,
                'subjectivity': 0.0,
                'emotions': {},
                'aspects': {},
                'word_count': 0
            }
        
        try:
            word_count = len(text.split())
            text_lower = text.lower()
            
            # Use TextBlob for polarity and subjectivity
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity  # -1 to 1
            subjectivity = blob.sentiment.subjectivity  # 0 to 1
            
            # Keyword-based fallback for short texts or when TextBlob is unreliable
            strong_pos = sum(1 for w in EnhancedSentimentAnalyzer.STRONG_POSITIVE if w in text_lower)
            strong_neg = sum(1 for w in EnhancedSentimentAnalyzer.STRONG_NEGATIVE if w in text_lower)
            weak_pos = sum(1 for w in EnhancedSentimentAnalyzer.WEAK_POSITIVE if w in text_lower)
            weak_neg = sum(1 for w in EnhancedSentimentAnalyzer.WEAK_NEGATIVE if w in text_lower)
            
            # Weighted keyword sentiment
            keyword_sentiment = (strong_pos * 2 + weak_pos) - (strong_neg * 2 + weak_neg)
            
            # For short texts or when TextBlob confidence is low, use keyword-based approach
            if word_count < 20 or (abs(polarity) < 0.2 and keyword_sentiment != 0):
                if keyword_sentiment > 0:
                    sentiment = 'positive'
                    confidence = min((strong_pos * 0.4 + weak_pos * 0.2) / max(word_count / 10, 1), 1.0)
                    if confidence < 0.1 and (strong_pos > 0 or weak_pos > 0):
                        confidence = 0.6
                elif keyword_sentiment < 0:
                    sentiment = 'negative'
                    confidence = min((strong_neg * 0.4 + weak_neg * 0.2) / max(word_count / 10, 1), 1.0)
                    if confidence < 0.1 and (strong_neg > 0 or weak_neg > 0):
                        confidence = 0.6
                else:
                    sentiment = 'neutral'
                    confidence = 1.0 - subjectivity
            else:
                # Determine sentiment from polarity (for longer texts)
                if polarity > 0.1:
                    sentiment = 'positive'
                    confidence = min(polarity, 1.0)
                elif polarity < -0.1:
                    sentiment = 'negative'
                    confidence = min(abs(polarity), 1.0)
                else:
                    sentiment = 'neutral'
                    confidence = max(1.0 - subjectivity, 0.5)
            
            # Detect emotions
            emotions = EmotionDetector.detect_emotions(text)
            dominant_emotion = EmotionDetector.get_dominant_emotion(emotions)
            
            # Analyze aspects
            aspects = AspectSentimentAnalyzer.analyze_all_aspects(text)
            
            return {
                'sentiment': sentiment,
                'confidence': float(max(confidence, 0.1)),  # Ensure minimum confidence
                'polarity': float(polarity),
                'subjectivity': float(subjectivity),
                'emotions': emotions,
                'dominant_emotion': dominant_emotion[0] if dominant_emotion else None,
                'aspects': aspects,
                'word_count': word_count
            }
        
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {e}")
            return {
                'sentiment': 'neutral',
                'confidence': 0.5,
                'polarity': 0.0,
                'subjectivity': 0.5,
                'emotions': {},
                'aspects': {},
                'word_count': len(text.split()),
                'error': str(e)
            }
