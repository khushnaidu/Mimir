import os
import logging
from typing import Dict, Any, List
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QueryProcessor:
    def __init__(self, model_name: str = "mistralai/Mistral-7B-v0.1"):
        """Initialize the QueryProcessor with Mistral model."""
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        
    def analyze_text(self, text: str) -> Dict:
        """
        Analyze the input text to extract components for News API and Reddit search.
        
        Args:
            text (str): The input text to analyze
            
        Returns:
            Dict containing:
                - news_api_components: Dict with topics, entities, timeframes, search_terms
                - reddit_query: Reformatted query optimized for semantic search
        """
        # Prepare prompt for the model
        prompt = f"""Analyze the following text and extract key components:
        Text: {text}
        
        Extract:
        1. Main topics
        2. Key entities (people, organizations, locations)
        3. Relevant timeframes
        4. Search terms for news articles
        5. Reformatted query for Reddit search
        
        Format the response as JSON."""
        
        # Generate analysis using Mistral
        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.model.generate(
            inputs["input_ids"],
            max_length=512,
            temperature=0.7,
            num_return_sequences=1
        )
        
        # Parse the model's response
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract components (simplified for now)
        news_api_components = {
            "topics": self._extract_topics(response),
            "entities": self._extract_entities(response),
            "timeframes": self._extract_timeframes(response),
            "search_terms": self._extract_search_terms(response)
        }
        
        reddit_query = self._format_reddit_query(response)
        
        return {
            "news_api_components": news_api_components,
            "reddit_query": reddit_query
        }
    
    def _extract_topics(self, response: str) -> List[str]:
        """Extract main topics from the model's response."""
        # Implementation would parse the response to extract topics
        return []
    
    def _extract_entities(self, response: str) -> List[str]:
        """Extract entities from the model's response."""
        # Implementation would parse the response to extract entities
        return []
    
    def _extract_timeframes(self, response: str) -> List[str]:
        """Extract timeframes from the model's response."""
        # Implementation would parse the response to extract timeframes
        return []
    
    def _extract_search_terms(self, response: str) -> List[str]:
        """Extract search terms from the model's response."""
        # Implementation would parse the response to extract search terms
        return []
    
    def _format_reddit_query(self, response: str) -> str:
        """Format the query for Reddit search while preserving context."""
        # Implementation would format the query based on the model's response
        return ""

    def process_query(self, text: str) -> Dict[str, Any]:
        """
        Process user's highlighted text and prepare it for both pipelines.
        
        Args:
            text: User's highlighted text
            
        Returns:
            Dictionary containing processed components for both pipelines
        """
        logger.info("Processing user query...")
        
        # Extract components for News API
        news_components = self._extract_news_components(text)
        
        # Format query for Reddit search
        reddit_query = self._format_reddit_query(text)
        
        return {
            'news_components': news_components,
            'reddit_query': reddit_query
        }
    
    def _extract_news_components(self, text: str) -> Dict[str, Any]:
        """
        Extract key components from text for News API search.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary containing extracted components
        """
        prompt = f"""Extract key components from this text for news search:
{text}

Return a JSON object with:
- main_topics: list of main topics
- entities: list of important entities (people, organizations, locations)
- timeframes: any mentioned dates or time periods
- search_terms: optimized search terms for news API

Format the response as a valid JSON object."""

        # Generate response
        response = self._generate_response(prompt)
        
        try:
            # Parse JSON response
            components = json.loads(response)
            logger.info(f"Extracted news components: {components}")
            return components
        except json.JSONDecodeError:
            logger.error("Failed to parse model response as JSON")
            return {
                'main_topics': [],
                'entities': [],
                'timeframes': [],
                'search_terms': text
            }
    
    def _generate_response(self, prompt: str) -> str:
        """
        Generate response using Mistral model.
        
        Args:
            prompt: Input prompt
            
        Returns:
            Generated response
        """
        # Format prompt for Mistral
        messages = [
            {"role": "system", "content": "You are a helpful assistant that processes text for political discourse analysis."},
            {"role": "user", "content": prompt}
        ]
        
        # Tokenize
        inputs = self.tokenizer.apply_chat_template(
            messages,
            return_tensors="pt"
        ).to(self.model.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_new_tokens=512,
                temperature=0.7,
                top_p=0.95,
                do_sample=True
            )
        
        # Decode response
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract assistant's response
        try:
            response = response.split("assistant\n")[-1].strip()
        except:
            response = response.strip()
        
        return response 