"""
LLM Pipeline Implementation with Multi-Model Support

This module provides functions to process political text using various LLM models,
including both commercial (OpenAI) and open-source models (Mistral).
"""
import os
import time
from typing import List, Dict, Any
import openai
import logging
from dotenv import load_dotenv
import torch
import platform

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")  # Get token from environment variable

# Log HF token status (without revealing the token)
if HF_TOKEN:
    logger.info("Hugging Face token is set")
else:
    logger.warning("Hugging Face token is not set. Open source models may not load correctly.")

# Available models configuration
OPENAI_MODELS = {
    "gpt-3.5-turbo": {"provider": "openai", "max_tokens": 512},
    "gpt-4": {"provider": "openai", "max_tokens": 512}
}

OPEN_SOURCE_MODELS = {
    "tinyllama-1.1b": {
        "provider": "huggingface",  # Use Hugging Face provider
        "model_id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",  # Changed to a smaller, open model
        "revision": "main",  # Use main branch to ensure compatibility
        "adapter_config": {  # Define adapter configuration for TinyLlama
            "use_fast_tokenizer": True,
            "use_auth_token": False,
            "low_cpu_mem_usage": True,
            "use_safetensors": True
        },
        "max_tokens": 512
    }
}

# Combine all models
AVAILABLE_MODELS = {**OPENAI_MODELS, **OPEN_SOURCE_MODELS}

# Set default model
DEFAULT_MODEL = "gpt-3.5-turbo"

# Task-specific optimized parameters
TASK_OPTIMIZED_PARAMS = {
    "query_reformatter": {
        "temperature": 0.2,  # Lower for more focused queries
        "top_p": 0.9,
        "frequency_penalty": 0.3  # Reduce repetition
    },
    "news_query_extractor": {
        "temperature": 0.3,
        "top_p": 0.85,
        "frequency_penalty": 0.5  # Higher to get diverse keywords
    },
    "context_summarizer": {
        "temperature": 0.6,  # Higher for more creative synthesis
        "top_p": 0.9,
        "presence_penalty": 0.5  # Encourage mentioning different aspects
    }
}

# Enhanced system prompts for political analysis
SYSTEM_PROMPTS = {
    "query_reformatter": "You are a political discourse expert who specializes in detecting key entities, policies, events and contextual relationships in political text. Focus on extracting substantive political concepts rather than rhetorical language.",
    
    "news_query_extractor": "You are a political journalist with deep knowledge of global politics, policies, and key political figures. Your specialty is identifying the most newsworthy and searchable aspects of political discussions.",
    
    "context_summarizer": "You are a balanced political analyst who provides nuanced perspectives across the political spectrum. You excel at synthesizing complex political discourse and presenting multiple viewpoints fairly."
}

# Few-shot examples for improved performance
FEW_SHOT_EXAMPLES = {
    "query_reformatter": [
        {"input": "I don't understand why the President is pushing this new healthcare policy", 
         "output": "Presidential healthcare policy reform current administration"},
        {"input": "The Senate vote on the infrastructure bill is coming up", 
         "output": "Senate infrastructure bill voting legislative process"}
    ],
    "news_query_extractor": [
        {"input": "The President announced new tariffs on Chinese goods", 
         "output": "Presidential tariffs, China trade policy, economic sanctions, bilateral trade relations"},
        {"input": "Supreme Court ruled 6-3 on the abortion case", 
         "output": "Supreme Court abortion ruling, constitutional law, judicial decision, reproductive rights"}
    ]
}

# Initialize OpenAI client
client = openai.OpenAI(api_key=OPENAI_API_KEY)

# Dictionary to cache Hugging Face models
hf_models = {}

class LLMQueryResult:
    """Class to store the result of an LLM query with performance metrics"""
    def __init__(self, content, model_name, latency, token_count=None):
        self.content = content
        self.model_name = model_name
        self.latency = latency  # in seconds
        self.token_count = token_count

def _format_few_shot_examples(task, user_content):
    """Format few-shot examples for a specific task"""
    if task not in FEW_SHOT_EXAMPLES or not FEW_SHOT_EXAMPLES[task]:
        return user_content
        
    examples = FEW_SHOT_EXAMPLES[task]
    examples_text = "\n\nExamples:\n"
    
    for example in examples:
        examples_text += f"Input: {example['input']}\nOutput: {example['output']}\n\n"
    
    # Add a clearer separator for the news_query_extractor to avoid confusion
    if task == "news_query_extractor":
        return f"{user_content}\n{examples_text}Now process the original input above, not the examples:"
    else:
        return f"{user_content}\n{examples_text}Now process this input:"

def call_model(model_name, messages, task_type=None, max_tokens=None, temperature=0.5):
    """
    Generic function to call any model with performance tracking
    
    Args:
        model_name: Name of the model to use
        messages: List of message dictionaries with 'role' and 'content'
        task_type: Type of task (for specialized parameters)
        max_tokens: Maximum tokens to generate
        temperature: Temperature for generation
        
    Returns:
        LLMQueryResult object with content and performance metrics
    """
    if model_name not in AVAILABLE_MODELS:
        logger.warning(f"Model {model_name} not found, using default model {DEFAULT_MODEL}")
        model_name = DEFAULT_MODEL
        
    model_config = AVAILABLE_MODELS[model_name]
    provider = model_config["provider"]
    
    # Apply task-specific parameters if available
    if task_type and task_type in TASK_OPTIMIZED_PARAMS:
        task_params = TASK_OPTIMIZED_PARAMS[task_type]
        temperature = task_params.get("temperature", temperature)
        top_p = task_params.get("top_p", 1.0)
        frequency_penalty = task_params.get("frequency_penalty", 0.0)
        presence_penalty = task_params.get("presence_penalty", 0.0)
    else:
        top_p = 1.0
        frequency_penalty = 0.0
        presence_penalty = 0.0
    
    start_time = time.time()
    content = ""
    token_count = None
    
    try:
        if provider == "openai":
            # Call OpenAI API
            if not max_tokens:
                max_tokens = model_config["max_tokens"]
                
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty
            )
            content = response.choices[0].message.content.strip()
            token_count = response.usage.total_tokens
            
        elif provider == "huggingface":
            # Use Hugging Face models
            model_id = model_config["model_id"]
            revision = model_config.get("revision", "main")  # Get revision or use "main" as default
            adapter_config = model_config.get("adapter_config", {})  # Get adapter config if available
            
            # Lazy loading of models
            if model_id not in hf_models:
                logger.info(f"Loading Hugging Face model: {model_id}")
                try:
                    from transformers import AutoModelForCausalLM, AutoTokenizer
                    
                    # Check if HF_TOKEN is available
                    if not HF_TOKEN:
                        logger.error("HF_TOKEN environment variable is not set. Cannot authenticate with Hugging Face.")
                        logger.info("Falling back to OpenAI model.")
                        # Fall back to OpenAI
                        fallback_messages = messages.copy()
                        fallback_response = client.chat.completions.create(
                            model="gpt-3.5-turbo",
                            messages=fallback_messages,
                            max_tokens=max_tokens or 512,
                            temperature=temperature
                        )
                        content = fallback_response.choices[0].message.content.strip()
                        token_count = fallback_response.usage.total_tokens
                        end_time = time.time()
                        latency = end_time - start_time
                        logger.info(f"Used OpenAI fallback for {model_name}")
                        return LLMQueryResult(content, "gpt-3.5-turbo (fallback)", latency, token_count)
                    
                    # Log attempt to load model with auth token
                    logger.info(f"Attempting to load model {model_id} with auth token")
                    
                    # Special handling for TinyLlama on macOS to fix tensor mismatch
                    use_safetensors = adapter_config.get("use_safetensors", True)  # Get from config or default to True
                    
                    # Log platform and configuration information
                    logger.info(f"Platform: {platform.system()}, Python: {platform.python_version()}, PyTorch: {torch.__version__}")
                    logger.info(f"Model adapter config: {adapter_config}")
                    
                    # For macOS specific diagnostics
                    if platform.system() == "Darwin":
                        logger.info(f"macOS version: {platform.mac_ver()[0]}")
                        if torch.backends.mps.is_available():
                            logger.info("MPS (Metal Performance Shaders) is available")
                        else:
                            logger.info("MPS is not available")
                    
                    # Load with appropriate parameters for memory efficiency
                    if "tinyllama" in model_id.lower() or "llama" in model_id.lower():
                        tokenizer = AutoTokenizer.from_pretrained(
                            model_id,
                            revision=revision,  # Use specified revision
                            trust_remote_code=True,
                            use_safetensors=use_safetensors  # Use safetensors format if specified
                        )
                    else:
                        tokenizer = AutoTokenizer.from_pretrained(
                            model_id,
                            token=HF_TOKEN,  # Updated from use_auth_token to token
                            revision=revision,  # Use specified revision
                            trust_remote_code=True,
                            use_safetensors=use_safetensors  # Use safetensors format if specified
                        )
                    
                    # For CPU usage, try to load a quantized version if available
                    if torch.cuda.is_available() and not (platform.system() == "Darwin" and "tinyllama" in model_id.lower()):
                        # Check if we're loading a Llama model which doesn't accept token parameter
                        if "tinyllama" in model_id.lower() or "llama" in model_id.lower():
                            model = AutoModelForCausalLM.from_pretrained(
                                model_id,
                                torch_dtype=torch.bfloat16,
                                device_map="auto",
                                revision=revision,  # Use specified revision
                                trust_remote_code=True
                            )
                        else:
                            model = AutoModelForCausalLM.from_pretrained(
                                model_id,
                                torch_dtype=torch.bfloat16,
                                device_map="auto",
                                token=HF_TOKEN,  # Updated from use_auth_token to token
                                revision=revision,  # Use specified revision
                                trust_remote_code=True
                            )
                    else:
                        # For CPU or macOS with TinyLlama, use CPU explicitly
                        logger.info(f"Using CPU for model loading (platform: {platform.system()})")
                        
                        # Check if we're loading a TinyLlama on macOS
                        if platform.system() == "Darwin" and ("tinyllama" in model_id.lower() or "llama" in model_id.lower()):
                            logger.info("Forcing CPU for TinyLlama on macOS to avoid MPS issues")
                            model = AutoModelForCausalLM.from_pretrained(
                                model_id,
                                device_map="cpu",
                                revision=revision,  # Use specified revision
                                trust_remote_code=True,
                                torch_dtype=torch.float32,  # Force float32 to avoid precision issues
                                use_safetensors=use_safetensors  # Use safetensors format if specified
                            )
                        else:
                            # For CPU, try to use a quantized model
                            try:
                                from transformers import BitsAndBytesConfig
                                
                                # 4-bit quantization for CPU
                                quantization_config = BitsAndBytesConfig(
                                    load_in_4bit=True,
                                    bnb_4bit_compute_dtype=torch.float32,
                                    bnb_4bit_quant_type="nf4",
                                    bnb_4bit_use_double_quant=True
                                )
                                
                                # Check if we're loading a Llama model which doesn't accept token parameter
                                if "tinyllama" in model_id.lower() or "llama" in model_id.lower():
                                    model = AutoModelForCausalLM.from_pretrained(
                                        model_id,
                                        quantization_config=quantization_config,
                                        device_map="auto",
                                        revision=revision,  # Use specified revision
                                        low_cpu_mem_usage=True,
                                        trust_remote_code=True
                                    )
                                else:
                                    model = AutoModelForCausalLM.from_pretrained(
                                        model_id,
                                        quantization_config=quantization_config,
                                        device_map="auto",
                                        low_cpu_mem_usage=True,
                                        token=HF_TOKEN,  # Updated from use_auth_token to token
                                        revision=revision,  # Use specified revision
                                        trust_remote_code=True
                                    )
                            except ImportError:
                                logger.warning("BitsAndBytes library not available. Attempting to load in 8-bit format.")
                                try:
                                    if "tinyllama" in model_id.lower() or "llama" in model_id.lower():
                                        # Force CPU for TinyLlama models on macOS to avoid MPS issues
                                        device_map = "cpu" if platform.system() == "Darwin" else "auto"
                                        model = AutoModelForCausalLM.from_pretrained(
                                            model_id,
                                            device_map=device_map,
                                            revision=revision,  # Use specified revision
                                            low_cpu_mem_usage=True,
                                            trust_remote_code=True
                                        )
                                    else:
                                        model = AutoModelForCausalLM.from_pretrained(
                                            model_id,
                                            device_map="auto",
                                            low_cpu_mem_usage=True,
                                            token=HF_TOKEN,
                                            revision=revision,  # Use specified revision
                                            trust_remote_code=True
                                        )
                                except Exception as quantization_error:
                                    logger.warning(f"8-bit loading failed: {str(quantization_error)}. Attempting to load in 4-bit format.")
                                    try:
                                        # Try a simpler loading approach
                                        if "tinyllama" in model_id.lower() or "llama" in model_id.lower():
                                            # Force CPU for TinyLlama models on macOS to avoid MPS issues
                                            device_map = "cpu" if platform.system() == "Darwin" else "auto"
                                            model = AutoModelForCausalLM.from_pretrained(
                                                model_id,
                                                device_map=device_map,
                                                revision=revision,  # Use specified revision
                                                low_cpu_mem_usage=True,
                                                trust_remote_code=True,
                                                torch_dtype=torch.float32  # Use float32 to avoid precision issues
                                            )
                                        else:
                                            model = AutoModelForCausalLM.from_pretrained(
                                                model_id,
                                                device_map="auto",
                                                low_cpu_mem_usage=True,
                                                token=HF_TOKEN,
                                                revision=revision,  # Use specified revision
                                                trust_remote_code=True
                                            )
                                    except Exception as basic_error:
                                        logger.error(f"Basic model loading failed: {str(basic_error)}. Falling back to OpenAI model.")
                                        # Fall back to OpenAI
                                        fallback_messages = messages.copy()
                                        fallback_response = client.chat.completions.create(
                                            model="gpt-3.5-turbo",
                                            messages=fallback_messages,
                                            max_tokens=max_tokens or 512,
                                            temperature=temperature
                                        )
                                        content = fallback_response.choices[0].message.content.strip()
                                        token_count = fallback_response.usage.total_tokens
                                        end_time = time.time()
                                        latency = end_time - start_time
                                        logger.info(f"Used OpenAI fallback for {model_name}")
                                        return LLMQueryResult(content, "gpt-3.5-turbo (fallback)", latency, token_count)
                    
                    hf_models[model_id] = {
                        "model": model,
                        "tokenizer": tokenizer
                    }
                    logger.info(f"Successfully loaded model: {model_id}")
                except Exception as e:
                    logger.error(f"Error loading model {model_id}: {str(e)}")
                    logger.info("Falling back to OpenAI model.")
                    # Fall back to OpenAI
                    fallback_messages = messages.copy()
                    fallback_response = client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=fallback_messages,
                        max_tokens=max_tokens or 512,
                        temperature=temperature
                    )
                    content = fallback_response.choices[0].message.content.strip()
                    token_count = fallback_response.usage.total_tokens
                    end_time = time.time()
                    latency = end_time - start_time
                    logger.info(f"Used OpenAI fallback for {model_name}")
                    return LLMQueryResult(content, "gpt-3.5-turbo (fallback)", latency, token_count)
            
            # Get cached model
            model = hf_models[model_id]["model"]
            tokenizer = hf_models[model_id]["tokenizer"]
            
            # Format messages for model-specific prompt format
            if "tinyllama" in model_id.lower():
                # Format for TinyLlama models
                prompt = ""
                for idx, msg in enumerate(messages):
                    role = msg["role"]
                    content_text = msg["content"]
                    
                    if role == "system" and idx == 0:
                        prompt += f"<|system|>\n{content_text}</s>\n"
                    elif role == "user":
                        prompt += f"<|user|>\n{content_text}</s>\n"
                    elif role == "assistant":
                        prompt += f"<|assistant|>\n{content_text}</s>\n"
                
                prompt += "<|assistant|>\n"
            
            elif "mistral" in model_id.lower():
                # Format for Mistral models
                prompt = ""
                for idx, msg in enumerate(messages):
                    role = msg["role"]
                    content_text = msg["content"]
                    
                    if role == "system" and idx == 0:
                        prompt += f"<|system|>\n{content_text}</s>\n"
                    elif role == "user":
                        prompt += f"<|user|>\n{content_text}</s>\n"
                    elif role == "assistant":
                        prompt += f"<|assistant|>\n{content_text}</s>\n"
                
                prompt += "<|assistant|>\n"
            
            else:
                # Generic format for other models
                prompt = ""
                for msg in messages:
                    role = msg["role"].capitalize()
                    content_text = msg["content"]
                    prompt += f"{role}: {content_text}\n"
                
                prompt += "Assistant: "
            
            # Generate with loaded model
            inputs = tokenizer(prompt, return_tensors="pt")
            
            # Check if we need to force CPU for TinyLlama on macOS
            if platform.system() == "Darwin" and "tinyllama" in model_id.lower():
                # Force CPU for TinyLlama on macOS to avoid MPS issues
                device = torch.device("cpu")
                # Make sure the model is instantiated first before moving to CPU
                if model is not None:
                    model = model.to(device)
                inputs = {k: v.to(device) for k, v in inputs.items()}
            else:
                # Use the model's current device if available
                if model is not None and hasattr(model, 'device'):
                    device = model.device
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                else:
                    # If model doesn't have a device, use CPU
                    device = torch.device("cpu")
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                
            max_new_tokens = max_tokens or model_config.get("max_tokens", 512)
            
            with torch.no_grad():
                outputs = model.generate(
                    inputs.input_ids,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=temperature > 0
                )
            
            full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the generated part (remove the prompt)
            content = full_output.replace(prompt, "").strip()
            token_count = len(outputs[0])
            
        elif provider == "ctransformers":
            # Use CTransformers for GGUF models (much more efficient on CPU)
            model_id = model_config["model_id"]
            filename = model_config.get("filename")
            
            # Lazy loading of models
            if model_id not in hf_models:
                logger.info(f"Loading CTransformers model: {model_id}")
                try:
                    # Import here to avoid dependencies if not using this provider
                    from transformers import AutoTokenizer
                    from ctransformers import AutoModelForCausalLM as CTModelForCausalLM
                    
                    # Get the model ID and filename for the GGUF file
                    if filename:
                        model_path = os.path.join(model_id, filename)
                    else:
                        model_path = model_id
                    
                    # Load tokenizer from Hugging Face
                    tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
                    
                    # Load model from GGUF file
                    model = CTModelForCausalLM.from_pretrained(
                        model_path,
                        model_type="llama",  # Updated from mistral to llama for TinyLlama
                        lib="avx2",  # CPU optimization level
                        context_length=2048,  # Reduce context length for CPU efficiency
                        gpu_layers=0  # No GPU
                    )
                    
                    hf_models[model_id] = {
                        "model": model,
                        "tokenizer": tokenizer
                    }
                    logger.info(f"Successfully loaded model: {model_id}")
                except Exception as e:
                    logger.error(f"Error loading model {model_id}: {str(e)}")
                    raise e
            
            # Get cached model
            model = hf_models[model_id]["model"]
            tokenizer = hf_models[model_id]["tokenizer"]
            
            # Format messages for Mistral
            prompt = ""
            for idx, msg in enumerate(messages):
                role = msg["role"]
                content_text = msg["content"]
                
                if role == "system" and idx == 0:
                    prompt += f"<|system|>\n{content_text}</s>\n"
                elif role == "user":
                    prompt += f"<|user|>\n{content_text}</s>\n"
                elif role == "assistant":
                    prompt += f"<|assistant|>\n{content_text}</s>\n"
            
            prompt += "<|assistant|>\n"
            
            # Generate with CTransformers
            max_new_tokens = max_tokens or model_config.get("max_tokens", 512)
            
            generated_text = model(
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=["</s>", "<|user|>"]  # Updated stop tokens for TinyLlama
            )
            
            # Extract only the generated part
            content = generated_text.replace(prompt, "").strip()
            token_count = len(content.split())  # Approximate
            
    except Exception as e:
        logger.error(f"Error calling model {model_name}: {str(e)}")
        content = f"Error processing request with {model_name}. Please try another model."
        
    end_time = time.time()
    latency = end_time - start_time
    
    logger.info(f"Model {model_name} ({task_type}) latency: {latency:.2f}s, tokens: {token_count}")
    
    return LLMQueryResult(content, model_name, latency, token_count)

def query_reformatter(text: str, model_name=DEFAULT_MODEL) -> LLMQueryResult:
    """
    Reformat user text for optimal semantic search in political context
    
    Args:
        text: User input text
        model_name: Model to use for reformatting
        
    Returns:
        LLMQueryResult with reformatted query
    """
    prompt = f"""Reformat the following text to be optimal for semantic search in a Reddit political discussion context.
    
IMPORTANT INSTRUCTIONS:
1. Preserve key political entities, topics, and issues
2. Extract main concepts and relationships
3. Remove filler words and unnecessary details
4. Structure as a concise query that would match relevant political discussions
5. Include key political terms that would appear in relevant posts
6. Don't make it too specific - aim for relevant results over exact matches

Original text: {text}
Reformatted query:"""
    
    # Add few-shot examples if available
    prompt = _format_few_shot_examples("query_reformatter", prompt)
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPTS["query_reformatter"]},
        {"role": "user", "content": prompt}
    ]
    
    return call_model(model_name, messages, task_type="query_reformatter", temperature=0.3)

def news_query_extractor(text: str, model_name=DEFAULT_MODEL) -> LLMQueryResult:
    """
    Extract keywords for news search from user text
    
    Args:
        text: User input text
        model_name: Model to use for extraction
        
    Returns:
        LLMQueryResult with extracted news queries
    """
    prompt = f"""Extract a simple, comma-separated list of keywords for news search about political topics.

IMPORTANT INSTRUCTIONS:
1. IF the input has political context (politicians, policies, laws, government actions), prioritize extracting those terms.
2. ALWAYS preserve the main entities and actions (WHO is doing WHAT).
3. Include specific policy areas, legislation names, or political events mentioned.
4. MAINTAIN the full context - don't omit important qualifiers or objects of actions.
5. Return 3-6 keywords/phrases that would help find RELEVANT news articles.
6. Format as a comma-separated list only, no explanations.
7. IGNORE ANY EXAMPLES AND ONLY PROCESS THE FOLLOWING TEXT.

Text to extract keywords from: {text}
Extracted keywords:"""
    
    # Add few-shot examples
    prompt = _format_few_shot_examples("news_query_extractor", prompt)
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPTS["news_query_extractor"]},
        {"role": "user", "content": prompt}
    ]
    
    return call_model(model_name, messages, task_type="news_query_extractor", temperature=0.3)

def context_summarizer(context: str, model_name=DEFAULT_MODEL) -> LLMQueryResult:
    """
    Summarize news and Reddit content for comprehensive political context
    
    Args:
        context: Raw context from news and Reddit
        model_name: Model to use for summarization
        
    Returns:
        LLMQueryResult with comprehensive summary
    """
    # Optimize context window if needed
    optimized_context = optimize_context_window(context)
    
    prompt = f"""Provide a comprehensive summary of the following context, highlighting different perspectives and key insights.
    
Your response MUST follow this exact structure:
1. Start with "Here are what other news sources are saying:" followed by a summary of the news articles.
2. Then include "Here is some relevant discourse on this and related issues on Reddit:" followed by a summary of the Reddit posts.

Context: {optimized_context}
Summary:"""
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPTS["context_summarizer"]},
        {"role": "user", "content": prompt}
    ]
    
    return call_model(model_name, messages, task_type="context_summarizer", temperature=0.5)

def optimize_context_window(context, max_context_length=6000):
    """
    Optimize context to fit within model's context window
    
    Args:
        context: Raw context string
        max_context_length: Maximum allowed context length
        
    Returns:
        Optimized context string
    """
    if len(context) <= max_context_length:
        return context
        
    # Simple approach: truncate to fit
    return context[:max_context_length] + "..."
    

