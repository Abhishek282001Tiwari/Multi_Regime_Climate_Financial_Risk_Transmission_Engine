"""
News Sentiment Analysis Module
Analyzes climate-related financial news sentiment.
"""

import pandas as pd
import numpy as np
import requests
from typing import Dict, List, Optional, Union
import logging
from datetime import datetime, timedelta
import re
from collections import Counter

class NewsAnalyzer:
    """Analyzes climate-related financial news sentiment."""
    
    def __init__(self, cache_dir: str = "data/cache"):
        self.cache_dir = cache_dir
        self.logger = logging.getLogger(__name__)
        
        # Simple sentiment lexicon
        self.positive_words = {
            'growth', 'increase', 'profit', 'gain', 'rise', 'boost', 'strong',
            'positive', 'success', 'opportunity', 'benefit', 'advance', 'improve',
            'renewable', 'sustainable', 'green', 'clean', 'innovation'
        }
        
        self.negative_words = {
            'decline', 'decrease', 'loss', 'fall', 'drop', 'weak', 'negative',
            'crisis', 'risk', 'threat', 'damage', 'disaster', 'concern',
            'carbon', 'emission', 'pollution', 'climate', 'extreme'
        }
        
        self.climate_keywords = {
            'climate', 'carbon', 'emission', 'renewable', 'green', 'sustainable',
            'environmental', 'esg', 'paris', 'agreement', 'temperature', 'warming',
            'flood', 'drought', 'hurricane', 'wildfire', 'sea level', 'melting'
        }
    
    def fetch_news_from_rss(self, rss_urls: List[str] = None) -> pd.DataFrame:
        """
        Fetch news from RSS feeds.
        
        Args:
            rss_urls: List of RSS feed URLs
            
        Returns:
            DataFrame with news articles
        """
        if rss_urls is None:
            rss_urls = [
                'https://feeds.reuters.com/reuters/environment',
                'https://feeds.bbci.co.uk/news/science_and_environment/rss.xml',
                'https://rss.cnn.com/rss/edition.rss'
            ]
        
        try:
            # For demonstration, simulate news articles
            # In production, you would parse actual RSS feeds
            
            articles = []
            for i in range(100):
                article = {
                    'title': f"Climate finance article {i+1}",
                    'content': self._generate_sample_article(),
                    'date': pd.Timestamp('2023-01-01') + pd.Timedelta(days=np.random.randint(0, 365)),
                    'source': np.random.choice(['Reuters', 'BBC', 'CNN']),
                    'url': f"https://example.com/article_{i+1}"
                }
                articles.append(article)
            
            return pd.DataFrame(articles)
            
        except Exception as e:
            self.logger.error(f"Error fetching news: {e}")
            return pd.DataFrame()
    
    def _generate_sample_article(self) -> str:
        """Generate sample article content for demonstration."""
        templates = [
            "Climate change impacts on financial markets show {} trends with {} concerns about {} risks.",
            "Green bonds market experiences {} growth as investors seek {} opportunities in {} sector.",
            "Carbon pricing mechanisms create {} pressure on {} industries with {} implications.",
            "Renewable energy investments show {} returns despite {} challenges in {} market.",
            "ESG funds attract {} capital flows as {} policies drive {} transformation."
        ]
        
        template = np.random.choice(templates)
        
        # Fill with random words
        words = ['significant', 'major', 'growing', 'declining', 'sustainable', 'renewable', 'carbon', 'climate']
        filled_template = template.format(*np.random.choice(words, 3))
        
        return filled_template
    
    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment of text using simple lexicon approach.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with sentiment scores
        """
        text = text.lower()
        words = re.findall(r'\b\w+\b', text)
        
        positive_count = sum(1 for word in words if word in self.positive_words)
        negative_count = sum(1 for word in words if word in self.negative_words)
        climate_count = sum(1 for word in words if word in self.climate_keywords)
        
        total_words = len(words)
        
        if total_words == 0:
            return {
                'sentiment_score': 0.0,
                'positive_ratio': 0.0,
                'negative_ratio': 0.0,
                'climate_relevance': 0.0
            }
        
        sentiment_score = (positive_count - negative_count) / total_words
        
        return {
            'sentiment_score': sentiment_score,
            'positive_ratio': positive_count / total_words,
            'negative_ratio': negative_count / total_words,
            'climate_relevance': climate_count / total_words
        }
    
    def process_news_sentiment(self, news_df: pd.DataFrame) -> pd.DataFrame:
        """
        Process news articles and calculate sentiment scores.
        
        Args:
            news_df: DataFrame with news articles
            
        Returns:
            DataFrame with sentiment analysis
        """
        results = []
        
        for _, row in news_df.iterrows():
            text = f"{row['title']} {row['content']}"
            sentiment = self.analyze_sentiment(text)
            
            result = {
                'date': row['date'],
                'source': row['source'],
                'title': row['title'],
                **sentiment
            }
            results.append(result)
        
        return pd.DataFrame(results)
    
    def aggregate_daily_sentiment(self, sentiment_df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate sentiment scores by day.
        
        Args:
            sentiment_df: DataFrame with sentiment scores
            
        Returns:
            DataFrame with daily aggregated sentiment
        """
        daily_sentiment = sentiment_df.groupby('date').agg({
            'sentiment_score': 'mean',
            'positive_ratio': 'mean',
            'negative_ratio': 'mean',
            'climate_relevance': 'mean'
        }).reset_index()
        
        return daily_sentiment
    
    def get_sentiment_indicators(self) -> pd.DataFrame:
        """
        Get comprehensive sentiment indicators.
        
        Returns:
            DataFrame with sentiment indicators
        """
        # Fetch news
        news_df = self.fetch_news_from_rss()
        
        # Process sentiment
        sentiment_df = self.process_news_sentiment(news_df)
        
        # Aggregate daily
        daily_sentiment = self.aggregate_daily_sentiment(sentiment_df)
        
        return daily_sentiment
