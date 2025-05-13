"""
LLM Pipeline Implementation

This module provides a function to synthesize a holistic summary from Reddit and News API results using an LLM.
"""
import os
from typing import List, Dict, Any
import openai

import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_NAME = "gpt-3.5-turbo"

client = openai.OpenAI(api_key=OPENAI_API_KEY)


def query_reformatter(text: str) -> str:
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
    
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "system", "content": "You are a political discourse analyst who helps format queries to find relevant political discussions."},
                  {"role": "user", "content": prompt}],
        max_tokens=128,
        temperature=0.3
    )
    return response.choices[0].message.content.strip()


def news_query_extractor(text: str) -> str:
    prompt = f"""Extract a simple, comma-separated list of keywords for news search about political topics.

IMPORTANT INSTRUCTIONS:
1. IF the input has political context (politicians, policies, laws, government actions), prioritize extracting those terms.
2. ALWAYS preserve the main entities and actions (WHO is doing WHAT).
3. Include specific policy areas, legislation names, or political events mentioned.
4. MAINTAIN the full context - don't omit important qualifiers or objects of actions.
5. Return 3-6 keywords/phrases that would help find RELEVANT news articles.
6. Format as a comma-separated list only, no explanations.

For example:
- "Trump passes new executive order to ban beef burgers" → "Trump executive order, beef burger ban, food regulation, presidential action"
- "Senate approves infrastructure bill with bipartisan support" → "Senate infrastructure bill, bipartisan support, congressional legislation, infrastructure investment"
- Remember, DO NOT just extract small keywords like "Trump" or "Biden" or random context words like "wish list" without the context of the political situation.
Text: {text}
Extracted keywords:"""
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "system", "content": "You are a political news analyst who extracts the most relevant search terms from headlines and stories."},
                  {"role": "user", "content": prompt}],
        max_tokens=128,
        temperature=0.3
    )
    return response.choices[0].message.content.strip()


def context_summarizer(context: str) -> str:
    prompt = f"""Provide a comprehensive summary of the following context, highlighting different perspectives and key insights.
    
Your response MUST follow this exact structure:
1. Start with "Here are what other news sources are saying:" followed by a summary of the news articles.
2. Then include "Here is some relevant discourse on this and related issues on Reddit:" followed by a summary of the Reddit posts.

Context: {context}
Summary:"""
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "system", "content": "You are a helpful assistant."},
                  {"role": "user", "content": prompt}],
        max_tokens=512,
        temperature=0.5
    )
    return response.choices[0].message.content.strip()
