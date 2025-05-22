# Core NLP Tasks in Mimir

This document details the three primary NLP tasks that form the backbone of Mimir's political analysis capabilities.

## Overview

Mimir uses specialized LLM tasks to process political text through a multi-stage pipeline:

1. **Query Reformulation**: Transforms user input into an optimized semantic search query for the Reddit vector store
2. **News Query Extraction**: Extracts focused keywords for retrieving relevant news articles
3. **Context Summarization**: Synthesizes Reddit posts and news articles into a balanced political analysis

These tasks work in concert to provide comprehensive political analysis with diverse perspectives.

## 1. Query Reformulation

### Purpose

Query reformulation optimizes the user's input text for semantic search against the vector store of Reddit political discussions. This transformation preserves key political concepts while structuring the query in a way that maximizes vector similarity matching.

### Implementation

```python
# From llm_pipeline.py
def query_reformatter(text: str, model_name=DEFAULT_MODEL) -> LLMQueryResult:
    """
    Reformat user text for optimal semantic search in political context
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
```

### Example

**Input**: 
```
I'm trying to understand why there's so much disagreement about immigration policy and what each side believes.
```

**Reformatted Query**: 
```
immigration policy disagreement partisan perspectives Republican Democrat border security economic impact pathways to citizenship
```

### Integration

The reformatted query is used in the RAG pipeline to search the vector store:

```python
# From rag_pipeline.py
async def process_query(self, user_text):
    # Step 1: Reformat the query for semantic search
    reformatted_query = self.llm_pipeline.query_reformatter(user_text)
    
    # Step 2: Search vector store for similar posts
    similar_posts = await self.search_vector_store(reformatted_query)
```

## 2. News Query Extraction

### Purpose

News query extraction transforms the user's input into focused keywords optimized for retrieving relevant news articles via the NewsAPI. These keywords need to be specific enough to find articles on the political topic but broad enough to capture diverse perspectives.

### Implementation

```python
# From llm_pipeline.py
def news_query_extractor(text: str, model_name=DEFAULT_MODEL) -> LLMQueryResult:
    """
    Extract keywords for news search from user text
    """
    prompt = f"""Extract a simple, comma-separated list of keywords for news search about political topics.

IMPORTANT INSTRUCTIONS:
1. IF the input has political context (politicians, policies, laws, government actions), prioritize extracting those terms.
2. ALWAYS preserve the main entities and actions (WHO is doing WHAT).
3. Include specific policy areas, legislation names, or political events mentioned.
4. MAINTAIN the full context - don't omit important qualifiers or objects of actions.
5. Return 3-6 keywords/phrases that would help find RELEVANT news articles.
6. Format as a comma-separated list only, no explanations.

Text to extract keywords from: {text}
Extracted keywords:"""
    
    # Add few-shot examples
    prompt = _format_few_shot_examples("news_query_extractor", prompt)
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPTS["news_query_extractor"]},
        {"role": "user", "content": prompt}
    ]
    
    return call_model(model_name, messages, task_type="news_query_extractor", temperature=0.3)
```

### Example

**Input**: 
```
What are the key policy differences between Democrats and Republicans on healthcare?
```

**Extracted Keywords**: 
```
healthcare policy differences, Democratic healthcare proposals, Republican healthcare proposals, Affordable Care Act, Medicare for All, private insurance
```

### Integration

The extracted keywords are processed and used to fetch news articles:

```python
# From main.py
news_query_result = llm_pipeline.news_query_extractor(text, model_name)
news_query_string = news_query_result.content

# Process news queries
if isinstance(news_query_string, str):
    raw_keywords = [k.strip() for k in news_query_string.split(',')]
    keywords = [k for k in raw_keywords if k]
    queries = []
    
    # Add multi-word phrases first
    for keyword in keywords:
        if len(keyword.split()) > 1:
            queries.append(keyword)
    
    # Add important single terms if needed
    if len(queries) < 3:
        political_entities = [k for k in keywords if len(k.split()) == 1 and len(k) > 3 
                             and k.lower() not in ['the', 'and', 'with', 'from', 'that', 'this']]
        queries.extend(political_entities[:3-len(queries)])
        
    # Ensure we have at least one query
    if not queries and keywords:
        queries = keywords[:3]
        
    # Add combined query for context
    if len(keywords) >= 2:
        main_terms = ' '.join(keywords[:3])
        if main_terms not in queries:
            queries.append(main_terms)

# Fetch news articles
news_results = await news_client.search_news(queries)
```

## 3. Context Summarization

### Purpose

Context summarization synthesizes the retrieved Reddit discussions and news articles into a comprehensive, balanced political analysis. This task involves:
- Integrating information from multiple sources
- Presenting diverse political perspectives
- Structuring information in a readable format
- Maintaining balance in political framing

### Implementation

```python
# From llm_pipeline.py
def context_summarizer(context: str, model_name=DEFAULT_MODEL) -> LLMQueryResult:
    """
    Summarize news and Reddit content for comprehensive political context
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
```

### Example

**Input Context**: (Truncated for brevity)
```
News Articles: [{'title': 'Democrats and Republicans Clash on Healthcare Priorities', 'description': 'Key differences in healthcare approaches...', ...}, ...]
Reddit Posts: [{'title': 'Healthcare Reform Discussion', 'text': 'What do you think about the current healthcare proposals?...', ...}, ...]
```

**Summarized Output**:
```
Here are what other news sources are saying: News coverage indicates that Democrats generally support expanding the Affordable Care Act (ACA) and potentially moving toward a public option or Medicare for All approach. They emphasize universal coverage, protecting pre-existing conditions, and controlling drug prices through government intervention. Republicans focus on market-based solutions, health savings accounts, price transparency, and reducing regulations. They advocate for greater competition among insurers and giving states more flexibility in managing healthcare programs.

Here is some relevant discourse on this and related issues on Reddit: Reddit discussions reflect similar partisan divides, with Democratic-leaning users emphasizing the moral imperative of universal coverage and Republican-leaning users concerned about government inefficiency and freedom of choice. Many centrist commenters note that both approaches have valid points but disagree on implementation. Several discussions mention that other developed nations have found ways to provide universal coverage while maintaining quality and controlling costs.
```

### Integration

The summarization is the final step in the analysis pipeline, providing the main output to the user:

```python
# From main.py
# Create context for summarization
context = (
    f"News Articles: {news_articles}\n"
    f"Reddit Posts: {reddit_posts}"
)

# Generate summary
summary_result = llm_pipeline.context_summarizer(context, model_name)
summary = summary_result.content

return {
    'summary': summary,
    'raw_context': {
        'reddit_posts': reddit_posts,
        'news_articles': news_articles
    },
    'model_used': model_name,
    # Other data...
}
```

## Model-Specific Optimizations

Each of these tasks has specific parameter optimizations:

```python
# From llm_pipeline.py
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
```

## Performance Metrics

The system collects the following performance metrics for each task:

1. **Latency**: Time taken to complete each task (in seconds)
2. **Token Count**: Number of tokens used for the task
3. **Total Process Time**: Overall time for the complete analysis pipeline

These metrics are tracked during normal operation and can help in:

- Identifying bottlenecks in the pipeline
- Comparing performance across different models
- Optimizing resource allocation

## Future Enhancements

Planned improvements for these core NLP tasks:

1. **Task-Specific Fine-Tuning**: Train specialized LoRA adapters for each task
2. **Cross-Model Distillation**: Distill knowledge from larger models into TinyLlama for each task
3. **Dynamic Parameter Selection**: Automatically adjust temperature and other parameters based on input complexity
4. **Multi-Model Ensemble**: Combine results from multiple models for improved quality
5. **User Feedback Integration**: Implement a working feedback mechanism to evaluate output quality 