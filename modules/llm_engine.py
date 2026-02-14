"""
Hybrid LLM integration - Ollama (local) and Gemini (cloud)
Provides review summarization, aspect extraction, and business insights
Uses the new google.genai package (recommended)
"""

import requests
import json
import logging
from typing import Dict, List, Optional
from dotenv import load_dotenv
import os

# Use new google.genai package
try:
    from google import genai
    GENAI_AVAILABLE = True
except ImportError:
    genai = None
    GENAI_AVAILABLE = False

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OllamaLLM:
    """Local Ollama LLM interface"""
    
    def __init__(self, model: str = "llama2", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url
        self.available = self._check_availability()
    
    def _check_availability(self) -> bool:
        """Check if Ollama is running"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=1)
            return response.status_code == 200
        except Exception as e:
            logger.debug(f"Ollama not available: {e}")
            return False
    
    def generate(self, prompt: str, temperature: float = 0.7) -> Optional[str]:
        """Generate text using Ollama"""
        if not self.available:
            return None
        
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "temperature": temperature,
                    "stream": False
                },
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json().get('response', '').strip()
            else:
                logger.error(f"Ollama error: {response.status_code}")
                return None
        except Exception as e:
            logger.error(f"Ollama generation failed: {e}")
            return None


class GeminiLLM:
    """Google Gemini LLM interface - uses new google.genai package"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        self.available = False
        self.client = None
        
        if self.api_key and genai and GENAI_AVAILABLE:
            try:
                self.client = genai.Client(api_key=self.api_key)
                self.available = True
                logger.info("Using google.genai package")
            except Exception as e:
                logger.warning(f"Gemini initialization failed: {e}")
    
    def generate(self, prompt: str, temperature: float = 0.7) -> Optional[str]:
        """Generate text using Gemini with new API"""
        if not self.available or not self.client:
            return None
        
        try:
            response = self.client.models.generate_content(
                model='gemini-2.0-flash-exp',
                contents=prompt,
                config=genai.types.GenerateContentConfig(
                    temperature=temperature,
                    max_output_tokens=1024
                )
            )
            return response.text.strip() if response.text else None
        except Exception as e:
            logger.error(f"Gemini generation failed: {e}")
            return None


class ReviewAnalysisEngine:
    """LLM-powered review analysis engine"""
    
    def __init__(self, prefer_local: bool = True):
        """
        Initialize with preference for local or cloud LLM
        prefer_local=True: Try Ollama first, fall back to Gemini
        prefer_local=False: Try Gemini first, fall back to Ollama
        """
        self.ollama = OllamaLLM()
        self.gemini = GeminiLLM()
        self.prefer_local = prefer_local
    
    def get_llm(self) -> Optional[object]:
        """Get available LLM based on preference"""
        if self.prefer_local:
            return self.ollama if self.ollama.available else self.gemini
        else:
            return self.gemini if self.gemini.available else self.ollama
    
    def summarize_reviews(self, reviews: List[str], max_length: int = 150) -> Optional[str]:
        """Summarize a batch of reviews"""
        llm = self.get_llm()
        if not llm:
            return None
        
        reviews_text = "\n".join([f"- {r}" for r in reviews[:10]])  # Limit to first 10
        
        prompt = f"""Summarize the following customer reviews in {max_length} words or less. 
Focus on key themes and overall sentiment.

Reviews:
{reviews_text}

Summary:"""
        
        return llm.generate(prompt, temperature=0.5)
    
    def extract_aspects(self, reviews: List[str]) -> Optional[Dict]:
        """Extract key aspects and their sentiment from reviews"""
        llm = self.get_llm()
        if not llm:
            return None
        
        reviews_text = "\n".join([f"- {r}" for r in reviews[:15]])
        
        prompt = f"""Analyze these customer reviews and identify the main aspects discussed.
For each aspect, indicate if the sentiment is mostly positive, negative, or mixed.

Return as JSON format:
{{"aspects": {{"aspect_name": "sentiment"}}}}

Reviews:
{reviews_text}

Analysis:"""
        
        response = llm.generate(prompt, temperature=0.3)
        
        try:
            # Try to extract JSON
            if response and '{' in response:
                json_start = response.find('{')
                json_end = response.rfind('}') + 1
                json_str = response[json_start:json_end]
                return json.loads(json_str)
        except Exception as e:
            logger.warning(f"JSON parsing failed: {e}")
        
        return None
    
    def generate_business_insights(self, reviews: List[str], ratings: List[float] = None) -> Optional[Dict]:
        """Generate actionable business insights from reviews"""
        llm = self.get_llm()
        
        reviews_text = "\n".join([f"- {r}" for r in reviews[:20]])
        rating_info = f"Average rating: {sum(ratings)/len(ratings):.1f}/5" if ratings else ""
        
        prompt = f"""Based on these customer reviews, provide business insights in JSON format:

{rating_info}

Reviews:
{reviews_text}

Provide JSON with these fields:
{{"strengths": ["..."], "weaknesses": ["..."], "improvement_suggestions": ["..."], "overall_recommendation": "..."}}

Analysis:"""
        
        response = None
        if llm:
            response = llm.generate(prompt, temperature=0.5)
        
        # If LLM generation failed or not available, use keyword-based fallback
        if not response:
            return ReviewAnalysisEngine._generate_fallback_insights(reviews, ratings)
        
        try:
            if response and '{' in response:
                json_start = response.find('{')
                json_end = response.rfind('}') + 1
                json_str = response[json_start:json_end]
                return json.loads(json_str)
        except Exception as e:
            logger.warning(f"JSON parsing failed: {e}, using fallback")
            return ReviewAnalysisEngine._generate_fallback_insights(reviews, ratings)
        
        return ReviewAnalysisEngine._generate_fallback_insights(reviews, ratings)
    
    def generate_improvement_recommendations(self, main_complaints: List[str]) -> Optional[str]:
        """Generate specific improvement recommendations"""
        llm = self.get_llm()
        
        if not main_complaints:
            return "No significant complaints found. Continue maintaining current quality standards."
        
        complaints_text = "\n".join([f"- {c}" for c in main_complaints])
        
        prompt = f"""Based on these customer complaints, provide 3-4 specific, actionable recommendations
to improve the product/service.

Main complaints:
{complaints_text}

Recommendations:"""
        
        if llm:
            response = llm.generate(prompt, temperature=0.7)
            if response:
                return response
        
        # Fallback: keyword-based recommendations
        return ReviewAnalysisEngine._generate_fallback_recommendations(main_complaints)
    
    def generate_marketing_angle(self, strengths: List[str]) -> Optional[str]:
        """Generate marketing angles based on product strengths"""
        llm = self.get_llm()
        
        if not strengths:
            return "Insufficient positive feedback to generate marketing angles."
        
        strengths_text = "\n".join([f"- {s}" for s in strengths])
        
        prompt = f"""Based on these product strengths mentioned by customers, 
suggest 2-3 marketing angles or unique selling propositions.

Customer-confirmed strengths:
{strengths_text}

Marketing angles:"""
        
        if llm:
            response = llm.generate(prompt, temperature=0.8)
            if response:
                return response
        
        # Fallback: keyword-based marketing angles
        return ReviewAnalysisEngine._generate_fallback_marketing_angle(strengths)
    
    @staticmethod
    def _generate_fallback_insights(reviews: List[str], ratings: List[float] = None) -> Dict:
        """Generate insights using keyword analysis when LLM unavailable"""
        positive_keywords = {'great', 'good', 'excellent', 'amazing', 'awesome', 'love', 'best', 'quality', 'fast', 'reliable'}
        negative_keywords = {'bad', 'poor', 'terrible', 'awful', 'worst', 'hate', 'issue', 'problem', 'late', 'broken'}
        
        strengths = set()
        weaknesses = set()
        
        for review in reviews:
            review_lower = review.lower()
            for word in positive_keywords:
                if word in review_lower:
                    strengths.add(word.capitalize())
            for word in negative_keywords:
                if word in review_lower:
                    weaknesses.add(word.capitalize())
        
        return {
            'strengths': list(strengths)[:5] if strengths else ['Product quality', 'Customer satisfaction'],
            'weaknesses': list(weaknesses)[:5] if weaknesses else ['Delivery time', 'Support response'],
            'improvement_suggestions': [
                'Address the top customer complaints through process improvements',
                'Enhance customer communication channels',
                'Implement quality assurance checks',
                'Consider customer feedback in product updates'
            ],
            'overall_recommendation': 'Continue monitoring customer feedback and focus on continuous improvement'
        }
    
    @staticmethod
    def _generate_fallback_recommendations(complaints: List[str]) -> str:
        """Generate fallback recommendations based on keywords"""
        recommendations = []
        complaint_text = " ".join(complaints).lower()
        
        if any(word in complaint_text for word in ['late', 'delay', 'delivery', 'shipping']):
            recommendations.append('• Optimize logistics and delivery processes to reduce delivery times')
        if any(word in complaint_text for word in ['broken', 'defect', 'quality', 'damaged']):
            recommendations.append('• Enhance quality control and packaging to reduce product damage')
        if any(word in complaint_text for word in ['support', 'service', 'help', 'response', 'customer care']):
            recommendations.append('• Improve customer support response times and availability')
        if any(word in complaint_text for word in ['price', 'expensive', 'cost', 'refund']):
            recommendations.append('• Review pricing strategy and refund policies')
        
        if not recommendations:
            recommendations = [
                '• Implement customer feedback system',
                '• Train team on customer service excellence',
                '• Regular product quality audits'
            ]
        
        return '\n'.join(recommendations[:4])
    
    @staticmethod
    def _generate_fallback_marketing_angle(strengths: List[str]) -> str:
        """Generate fallback marketing angles"""
        angles = []
        strengths_text = " ".join(strengths).lower()
        
        if any(word in strengths_text for word in ['quality', 'excellent', 'good']):
            angles.append('• Premium quality assurance and customer-verified excellence')
        if any(word in strengths_text for word in ['fast', 'quick', 'delivery']):
            angles.append('• Speed and reliability - trusted for quick delivery')
        if any(word in strengths_text for word in ['price', 'affordable', 'value']):
            angles.append('• Best value for money with premium quality')
        if any(word in strengths_text for word in ['service', 'support', 'help']):
            angles.append('• Exceptional customer support and care')
        
        if not angles:
            angles = [
                '• Customer-proven reliability and satisfaction',
                '• Award-winning quality and service'
            ]
        
        return '\n'.join(angles[:3])


class InsightAggregator:
    """Aggregate and format insights for dashboard display"""
    
    @staticmethod
    def aggregate_aspects(aspect_sentiments: List[Dict]) -> Dict[str, Dict]:
        """Aggregate aspect sentiments across reviews"""
        aspects = {}
        
        for sentiment_data in aspect_sentiments:
            for aspect, data in sentiment_data.get('aspects', {}).items():
                if aspect not in aspects:
                    aspects[aspect] = {
                        'positive': 0, 'negative': 0, 'neutral': 0,
                        'total': 0, 'confidence_sum': 0
                    }
                
                sentiment = data.get('sentiment', 'neutral')
                aspects[aspect][sentiment] += 1
                aspects[aspect]['total'] += 1
                aspects[aspect]['confidence_sum'] += data.get('confidence', 0.5)
        
        # Calculate percentages
        for aspect, stats in aspects.items():
            if stats['total'] > 0:
                stats['positive_pct'] = (stats['positive'] / stats['total']) * 100
                stats['negative_pct'] = (stats['negative'] / stats['total']) * 100
                stats['neutral_pct'] = (stats['neutral'] / stats['total']) * 100
                stats['avg_confidence'] = stats['confidence_sum'] / stats['total']
        
        return aspects
    
    @staticmethod
    def aggregate_emotions(emotion_lists: List[Dict]) -> Dict[str, float]:
        """Aggregate emotions across reviews"""
        emotion_totals = {}
        
        for emotions in emotion_lists:
            for emotion, confidence in emotions.items():
                emotion_totals[emotion] = emotion_totals.get(emotion, 0) + confidence
        
        if emotion_lists:
            emotion_totals = {e: total / len(emotion_lists) for e, total in emotion_totals.items()}
        
        return emotion_totals
