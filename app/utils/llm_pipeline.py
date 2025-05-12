"""
LLM Pipeline Implementation

This module provides a function to synthesize a holistic summary from Reddit and News API results using an LLM.
"""
import os
from typing import List, Dict, Any
import openai  # Or replace with your preferred LLM API

# You may want to load your API key from an environment variable or config file
OPENAI_API_KEY = "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"

# Set your model name (e.g., 'gpt-3.5-turbo' or 'gpt-4')
MODEL_NAME = "gpt-3.5-turbo"

client = openai.OpenAI(api_key=OPENAI_API_KEY)

def query_reformatter(text: str) -> str:
    prompt = f"""Reformat the following text to be optimal for semantic search while preserving key concepts.\nOriginal text: {text}\nReformatted query:"""
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "system", "content": "You are a helpful assistant."},
                  {"role": "user", "content": prompt}],
        max_tokens=128,
        temperature=0.3
    )
    return response.choices[0].message.content.strip()

def news_query_extractor(text: str) -> str:
    prompt = f"""Extract a simple, comma-separated list of keywords for news search. Only output the keywords, nothing else.\nText: {text}\nExtracted components:"""
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "system", "content": "You are a helpful assistant."},
                  {"role": "user", "content": prompt}],
        max_tokens=128,
        temperature=0.3
    )
    return response.choices[0].message.content.strip()

def context_summarizer(context: str) -> str:
    prompt = f"""Provide a comprehensive summary of the following context, highlighting different perspectives and key insights.\nContext: {context}\nSummary:"""
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "system", "content": "You are a helpful assistant."},
                  {"role": "user", "content": prompt}],
        max_tokens=256,
        temperature=0.5
    )
    return response.choices[0].message.content.strip()
