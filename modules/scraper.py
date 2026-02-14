"""
Simplified and robust web scraper for e-commerce product reviews
Works across Amazon, Flipkart, and generic e-commerce platforms
"""

import time
import re
import requests
from bs4 import BeautifulSoup
from typing import List, Dict, Optional
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Simple review validation - more lenient to catch more reviews
def is_valid_review(text: str) -> bool:
    """Check if text is a valid customer review"""
    if not text:
        return False

    text_clean = text.strip()
    if len(text_clean) < 10:  # Reduced from 20 to catch short reviews
        return False

    # Reject obvious non-reviews (prices, specs, navigation)
    blacklist = [
        r"\₹\s*\d+",
        r"\$\s*\d+",
        r"\bgb\b|\bmah\b|\bmp\b|\bwatt\b",
        r"add to cart|buy now|available|checkout",
        r"specification|specifications|features|highlights|description",
        r"sign up|login|log in|register|account",
        r"terms|privacy|policy|cookie",
        r"^(loading|please wait|processing)",
        r"^(you may also like|customers who bought|related items)"
    ]

    lower_text = text_clean.lower()
    for pattern in blacklist:
        if re.search(pattern, lower_text):
            return False

    # Accept if contains opinion/sentiment words (expanded list)
    opinion_words = [
        "good", "bad", "great", "excellent", "poor", "amazing", "awesome",
        "love", "hate", "disappointed", "satisfied", "happy", "unhappy",
        "worth", "recommend", "quality", "issue", "problem", "perfect",
        "terrible", "horrible", "fast", "slow", "better", "worst", "best",
        "durable", "broken", "defective", "beautiful", "ugly", "nice", "ok",
        "okay", "decent", "fair", "average", "wonderful", "fantastic",
        "disappointing", "waste", "using", "use", "used", "works", "work",
        "delivered", "delivery", "arrived", "received", "packed", "product",
        "value", "price", "expensive", "cheap", "battery", "performance",
        "camera", "screen", "display", "sound", "better than", "worse than"
    ]

    # Check for opinion words
    if any(word in lower_text for word in opinion_words):
        return True
    
    # Accept longer text (30+ chars) even without explicit opinion words
    if len(text_clean) >= 30:
        return True
    
    return False


# Amazon scraper (using review pages)
def scrape_amazon(url: str, max_reviews=50) -> List[Dict]:
    """Scrape reviews from Amazon product page"""
    reviews = []
    
    try:
        # Extract ASIN from URL
        if "/dp/" not in url:
            logger.warning("Not an Amazon product URL")
            return reviews
            
        base = url.split("/dp/")[0]
        asin = url.split("/dp/")[-1].split("/")[0][:10]
        review_url = f"{base}/product-reviews/{asin}"
        
        logger.info(f"Scraping Amazon reviews from: {review_url}")
        
        # Try multiple pages - go up to page 5 to ensure we get enough reviews
        for page in range(1, 6):
            page_url = f"{review_url}?pageNumber={page}"
            
            try:
                response = requests.get(page_url, timeout=10)
                soup = BeautifulSoup(response.text, "html.parser")
                
                # Find review blocks
                blocks = soup.select("div[data-hook='review']")
                logger.info(f"Page {page}: Found {len(blocks)} review blocks")
                
                if not blocks:
                    break
                
                for block in blocks:
                    # Get review text
                    text_elem = block.select_one("span[data-hook='review-body']")
                    if text_elem:
                        review_text = text_elem.get_text(strip=True)
                        if is_valid_review(review_text):
                            reviews.append({
                                "text": review_text[:500],
                                "source": "amazon"
                            })
                            logger.debug(f"Found review: {review_text[:60]}...")
                
                if len(reviews) >= max_reviews:
                    break
                    
            except Exception as e:
                logger.debug(f"Error on page {page}: {e}")
                continue
        
        logger.info(f"Amazon: Extracted {len(reviews)} reviews")
        return reviews
        
    except Exception as e:
        logger.error(f"Amazon scraper error: {e}")
        return reviews


# Flipkart scraper
def scrape_flipkart(url: str, max_reviews=50) -> List[Dict]:
    """Scrape reviews from Flipkart product page"""
    reviews = []
    
    try:
        logger.info(f"Scraping Flipkart reviews from: {url}")
        
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")
        
        # Flipkart review selectors
        review_blocks = soup.select("div.t-ZTKy, div.ZmyHeo, div._6K-7Co")
        logger.info(f"Found {len(review_blocks)} review blocks")
        
        for block in review_blocks:
            text = block.get_text(strip=True)
            if is_valid_review(text):
                reviews.append({
                    "text": text[:500],
                    "source": "flipkart"
                })
                logger.debug(f"Found review: {text[:60]}...")
            
            if len(reviews) >= max_reviews:
                break
        
        logger.info(f"Flipkart: Extracted {len(reviews)} reviews")
        return reviews
        
    except Exception as e:
        logger.error(f"Flipkart scraper error: {e}")
        return reviews


# Generic Selenium scraper for dynamic content
def scrape_generic_selenium(url: str, max_reviews=50) -> List[Dict]:
    """Generic Selenium-based scraper for any e-commerce site"""
    reviews = []
    driver = None
    
    try:
        logger.info(f"Starting Selenium scraper for: {url}")
        
        # Setup Chrome options
        options = Options()
        options.add_argument("--headless")
        options.add_argument("--disable-gpu")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        
        driver = webdriver.Chrome(options=options)
        driver.get(url)
        
        logger.info(f"Page title: {driver.title}")
        time.sleep(5)
        
        # Scroll to load reviews - more aggressive scrolling
        for i in range(10):
            logger.debug(f"Scrolling page {i+1}...")
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight)")
            time.sleep(1.5)
        
        # Find review elements - more selectors to catch all reviews
        review_elements = driver.find_elements(By.XPATH, 
            "//div[@role='article'] | //div[contains(@class, 'review')] | //div[contains(@class, 'comment')] | //p[text()] | //span[contains(@class, 'text')]"
        )
        
        logger.info(f"Found {len(review_elements)} potential review elements")
        
        checked = 0
        for element in review_elements:
            try:
                text = element.text.strip()
                if text and len(text) > 5:  # Even check very short text
                    checked += 1
                    
                    if is_valid_review(text):
                        reviews.append({
                            "text": text[:500],
                            "source": "generic"
                        })
                        logger.debug(f"Found review: {text[:60]}...")
                    
                    if len(reviews) >= max_reviews:
                        break
                    
            except Exception as e:
                logger.debug(f"Error processing element: {e}")
                continue
        
        logger.info(f"Selenium: Checked {checked} elements, found {len(reviews)} reviews")
        
    except Exception as e:
        logger.error(f"Selenium scraper error: {e}")
        
    finally:
        if driver:
            try:
                driver.quit()
            except:
                pass
    
    return reviews


# ReviewScraper class for app.py compatibility
class ReviewScraper:
    """Unified review scraper interface"""
    
    def __init__(self):
        """Initialize the scraper"""
        logger.info("ReviewScraper initialized")
    
    def scrape_reviews(self, url: str, max_reviews: int = 50) -> Optional[List[Dict]]:
        """
        Main method to scrape reviews from various platforms
        
        Args:
            url: Product URL
            max_reviews: Maximum number of reviews to extract
            
        Returns:
            List of review dictionaries with 'text' and 'source' keys
        """
        if not url:
            logger.error("No URL provided")
            return None
        
        url_lower = url.lower()
        reviews = []
        
        try:
            # Platform detection
            if "amazon" in url_lower:
                logger.info("Detected Amazon platform")
                reviews = scrape_amazon(url, max_reviews)
                
            elif "flipkart" in url_lower:
                logger.info("Detected Flipkart platform")
                reviews = scrape_flipkart(url, max_reviews)
                
            else:
                logger.info("Using generic Selenium scraper")
                reviews = scrape_generic_selenium(url, max_reviews)
            
            # If no reviews found, try Selenium as fallback
            if not reviews or len(reviews) < 10:  # Changed from 3 to 10
                logger.info(f"Insufficient reviews found ({len(reviews) if reviews else 0}), trying Selenium fallback")
                selenium_reviews = scrape_generic_selenium(url, max_reviews)
                if selenium_reviews and len(selenium_reviews) > len(reviews or []):
                    reviews = selenium_reviews
            
            # Deduplicate reviews
            if reviews:
                seen = set()
                unique_reviews = []
                for r in reviews:
                    key = r['text'][:100]
                    if key not in seen:
                        seen.add(key)
                        unique_reviews.append(r)
                
                logger.info(f"After deduplication: {len(unique_reviews)} unique reviews")
                return unique_reviews if unique_reviews else None
            
            logger.warning("No reviews found")
            return None
            
        except Exception as e:
            logger.error(f"Scraping error: {e}")
            return None
