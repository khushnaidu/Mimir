"""
Political Analysis Module using TinyLlama with LoRA

This module demonstrates how the optimized TinyLlama model with LoRA adapters
can be used for political text analysis tasks in the Mimir project.
"""
import sys
import os
import logging
from typing import Dict, List, Any
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Import from LLM pipeline
from .llm_pipeline import (
    call_model, 
    OPEN_SOURCE_MODELS, 
    LLMQueryResult
)

class PoliticalAnalyzer:
    """
    A class that provides political text analysis using TinyLlama with LoRA adapters.
    
    This demonstrates how the optimized model can be used for various political
    analysis tasks, showcasing the benefits of the parameter-efficient fine-tuning.
    """
    
    def __init__(self, model_name="tinyllama-1.1b"):
        """Initialize the analyzer with the specified model"""
        self.model_name = model_name
        logger.info(f"Initializing PoliticalAnalyzer with {model_name}")
        # Ensure LoRA is enabled
        if model_name in OPEN_SOURCE_MODELS:
            OPEN_SOURCE_MODELS[model_name]["adapter_config"]["use_peft"] = True
    
    def analyze_policy_differences(self, topic: str) -> LLMQueryResult:
        """
        Analyze policy differences between political parties on a specific topic
        
        Args:
            topic: The political topic to analyze (e.g., "healthcare", "climate change")
            
        Returns:
            LLMQueryResult containing the analysis
        """
        logger.info(f"Analyzing policy differences on {topic}")
        
        # Construct prompt for policy difference analysis
        messages = [
            {"role": "system", "content": "You are a balanced political analyst specializing in identifying policy differences between political parties."},
            {"role": "user", "content": f"What are the key policy differences between Democrats and Republicans on {topic}? Provide a balanced analysis that considers each party's perspective."}
        ]
        
        return call_model(self.model_name, messages, task_type="context_summarizer")
    
    def identify_bipartisan_opportunities(self, topic: str) -> LLMQueryResult:
        """
        Identify potential bipartisan opportunities on a specific topic
        
        Args:
            topic: The political topic to analyze (e.g., "infrastructure", "education")
            
        Returns:
            LLMQueryResult containing the analysis
        """
        logger.info(f"Identifying bipartisan opportunities on {topic}")
        
        # Construct prompt for bipartisan opportunity analysis
        messages = [
            {"role": "system", "content": "You are a political analyst who specializes in finding common ground between opposing political viewpoints."},
            {"role": "user", "content": f"What are the potential areas for bipartisan agreement on {topic}? Identify policy aspects where Democrats and Republicans might find common ground."}
        ]
        
        return call_model(self.model_name, messages, task_type="context_summarizer")
    
    def analyze_political_speech(self, speech_text: str) -> LLMQueryResult:
        """
        Analyze a political speech or statement
        
        Args:
            speech_text: The text of the political speech to analyze
            
        Returns:
            LLMQueryResult containing the analysis
        """
        logger.info("Analyzing political speech")
        
        # Construct prompt for speech analysis
        messages = [
            {"role": "system", "content": "You are a political discourse analyst who provides balanced analysis of political rhetoric and language."},
            {"role": "user", "content": f"Analyze the following political statement, identifying key themes, policy positions, and rhetorical strategies:\n\n{speech_text}"}
        ]
        
        return call_model(self.model_name, messages, task_type="context_summarizer")
    
    def extract_policy_keywords(self, policy_text: str) -> LLMQueryResult:
        """
        Extract key policy terms and concepts from text
        
        Args:
            policy_text: Text containing policy information
            
        Returns:
            LLMQueryResult containing extracted keywords
        """
        logger.info("Extracting policy keywords")
        
        # Construct prompt for keyword extraction
        messages = [
            {"role": "system", "content": "You are a political terminology expert who identifies key policy terms and concepts."},
            {"role": "user", "content": f"Extract keywords from the following text for news search. IMPORTANT: Return ONLY a comma-separated list of 5-8 specific keywords or short phrases, with NO additional text, explanation, or analysis:\n\n{policy_text}"}
        ]
        
        # Get the raw result
        result = call_model(self.model_name, messages, task_type="news_query_extractor")
        
        # Post-process to remove any instructions that might have leaked into the response
        if result.content:
            # Define phrases to filter out (from instructions that might get repeated in output)
            instruction_phrases = [
                "important", "return only", "comma-separated list", "specific keywords",
                "no additional text", "explanation", "analysis"
            ]
            
            # Split by commas and clean each keyword
            raw_keywords = [k.strip() for k in result.content.split(',')]
            
            # Filter out instruction phrases
            cleaned_keywords = []
            for keyword in raw_keywords:
                # Skip empty keywords
                if not keyword:
                    continue
                    
                # Skip keywords that look like instruction phrases
                if any(phrase.lower() in keyword.lower() for phrase in instruction_phrases):
                    continue
                    
                cleaned_keywords.append(keyword)
            
            # If we filtered out too many keywords, ensure we have some basic political keywords
            if len(cleaned_keywords) < 3:
                # Extract obvious entities from the text
                import re
                entities = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', policy_text)
                if entities:
                    cleaned_keywords.extend(entities[:3])
                
                # Add some generic political keywords if needed
                generic_keywords = ["international relations", "policy analysis", "political context"]
                for keyword in generic_keywords:
                    if len(cleaned_keywords) < 5:
                        cleaned_keywords.append(keyword)
            
            # Join back with commas
            result.content = ", ".join(cleaned_keywords)
        
        return result

# Demo function to show the analyzer in action
def run_political_analysis_demo():
    """Run a demonstration of the PoliticalAnalyzer capabilities"""
    analyzer = PoliticalAnalyzer()
    
    print("\n=== MIMIR POLITICAL ANALYSIS DEMO ===")
    print("Using TinyLlama with LoRA parameter-efficient adaptation\n")
    
    # Demo 1: Policy differences on healthcare
    result = analyzer.analyze_policy_differences("healthcare")
    print("\n--- POLICY DIFFERENCES ANALYSIS: HEALTHCARE ---")
    print(result.content)
    print(f"Response time: {result.latency:.2f}s")
    
    # Demo 2: Bipartisan opportunities on infrastructure
    result = analyzer.identify_bipartisan_opportunities("infrastructure")
    print("\n--- BIPARTISAN OPPORTUNITIES: INFRASTRUCTURE ---")
    print(result.content)
    print(f"Response time: {result.latency:.2f}s")
    
    # Demo 3: Political speech analysis
    speech = """
    We must work together to address the pressing challenges of our time. 
    Our administration is committed to creating jobs through infrastructure investment,
    protecting the environment with renewable energy, and ensuring healthcare
    is accessible to all Americans. We will also strengthen our position on the global stage.
    """
    result = analyzer.analyze_political_speech(speech)
    print("\n--- POLITICAL SPEECH ANALYSIS ---")
    print(result.content)
    print(f"Response time: {result.latency:.2f}s")
    
    print("\n=== DEMO COMPLETE ===")

if __name__ == "__main__":
    run_political_analysis_demo() 