"""
Test script for TinyLlama implementation with LoRA adapters
Demonstrates how to use the lightweight TinyLlama model with PEFT
"""
import os
import sys
import logging
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Import from LLM pipeline
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.llm_pipeline import call_model, OPEN_SOURCE_MODELS, hf_models

def test_tinyllama():
    """Test TinyLlama model with and without LoRA adapters"""
    logger.info("Testing TinyLlama model with lightweight implementation")
    
    # Define test messages
    test_messages = [
        {"role": "system", "content": "You are a political discourse analyzer that helps provide balanced analysis of political topics."},
        {"role": "user", "content": "What are the key policy differences between Democrats and Republicans on healthcare?"}
    ]
    
    # Call the model
    model_name = "tinyllama-1.1b"
    logger.info(f"Calling {model_name} with test messages")
    
    # Test with LoRA adapter first
    logger.info("Testing With LoRA Adapter")
    # Ensure adapter is enabled
    OPEN_SOURCE_MODELS[model_name]["adapter_config"]["use_peft"] = True
    
    # Call model with adapter
    result_with_adapter = call_model(model_name, test_messages, task_type="context_summarizer")
    
    # Log the result
    logger.info(f"Result from With LoRA Adapter:")
    logger.info(f"Content: {result_with_adapter.content}")
    logger.info(f"Latency: {result_with_adapter.latency:.2f}s")
    logger.info(f"Token count: {result_with_adapter.token_count}")
    
    # Clear previously loaded model to force reload without adapter
    model_id = OPEN_SOURCE_MODELS[model_name]["model_id"]
    if model_id in hf_models:
        generator = hf_models[model_id]["generator"]
        # If it's our custom model, explicitly remove the adapter
        if hasattr(generator, 'remove_lora_adapter'):
            generator.remove_lora_adapter()
        # Delete the model from cache to force reload
        del hf_models[model_id]
    
    # Test without adapter
    logger.info("Testing Without Adapter (Baseline)")
    # Disable adapter
    OPEN_SOURCE_MODELS[model_name]["adapter_config"]["use_peft"] = False
    
    # Call model without adapter
    result_without_adapter = call_model(model_name, test_messages, task_type="context_summarizer")
    
    # Log the result
    logger.info(f"Result from Without Adapter (Baseline):")
    logger.info(f"Content: {result_without_adapter.content}")
    logger.info(f"Latency: {result_without_adapter.latency:.2f}s")
    logger.info(f"Token count: {result_without_adapter.token_count}")
    
    # Restore original config
    OPEN_SOURCE_MODELS[model_name]["adapter_config"]["use_peft"] = True
        
    logger.info("Testing complete")
    
if __name__ == "__main__":
    test_tinyllama() 