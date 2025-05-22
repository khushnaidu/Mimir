# Mimir API Documentation

This document details the REST API endpoints available in the Mimir political analysis tool.

## Base URL

All API endpoints are relative to the base URL:

```
http://localhost:5000
```

## Authentication

Currently, the API does not implement authentication. It is recommended to run the server locally or behind a secure proxy if deploying to production.

## Endpoints

### Get Available Models

Retrieve a list of available language models for analysis.

**Endpoint**: `GET /models`

**Response**:
```json
{
  "models": ["gpt-3.5-turbo", "gpt-4", "tinyllama-1.1b"],
  "default_model": "gpt-3.5-turbo",
  "model_descriptions": {
    "gpt-3.5-turbo": "OpenAI GPT-3.5 Turbo - Fast and efficient commercial model",
    "gpt-4": "OpenAI GPT-4 - Advanced commercial model with strong reasoning",
    "tinyllama-1.1b": "TinyLlama with LoRA - Open-source model optimized for political analysis"
  }
}
```

### Analyze Political Text

Analyze political text using the specified model.

**Endpoint**: `POST /analyze`

**Request Body**:
```json
{
  "text": "What are the key policy differences between Democrats and Republicans on healthcare?",
  "model": "tinyllama-1.1b",
  "collect_feedback": true
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| text | string | Yes | The political text to analyze |
| model | string | No | The model to use (defaults to "gpt-3.5-turbo") |
| collect_feedback | boolean | No | Whether to collect feedback for evaluation (defaults to true) |

**Response**:
```json
{
  "summary": "Here are what other news sources are saying: News coverage indicates that Democrats generally support expanding the Affordable Care Act (ACA) and potentially moving toward a public option or Medicare for All approach. They emphasize universal coverage, protecting pre-existing conditions, and controlling drug prices through government intervention. Republicans focus on market-based solutions, health savings accounts, price transparency, and reducing regulations. They advocate for greater competition among insurers and giving states more flexibility in managing healthcare programs.\n\nHere is some relevant discourse on this and related issues on Reddit: Reddit discussions reflect similar partisan divides, with Democratic-leaning users emphasizing the moral imperative of universal coverage and Republican-leaning users concerned about government inefficiency and freedom of choice. Many centrist commenters note that both approaches have valid points but disagree on implementation. Several discussions mention that other developed nations have found ways to provide universal coverage while maintaining quality and controlling costs.",
  "raw_context": {
    "reddit_posts": [
      {
        "id": "abc123",
        "title": "Healthcare Reform Discussion",
        "text": "What do you think about the current healthcare proposals?...",
        "subreddit": "r/PoliticalDiscussion",
        "score": 245,
        "similarity_score": 0.89
      }
    ],
    "news_articles": [
      {
        "title": "Democrats and Republicans Clash on Healthcare Priorities",
        "description": "Key differences in healthcare approaches...",
        "url": "https://example.com/news/healthcare-debate",
        "publishedAt": "2023-05-15T14:30:00Z",
        "source": {
          "name": "Example News"
        }
      }
    ]
  },
  "model_used": "tinyllama-1.1b",
  "query_id": "550e8400-e29b-41d4-a716-446655440000",
  "performance_metrics": {
    "total_process_time": 4.32,
    "reformatting_time": 0.78,
    "news_query_time": 0.85,
    "summarization_time": 2.69
  }
}
```

### Submit Feedback

Submit user feedback on analysis results.

**Endpoint**: `POST /feedback`

**Request Body**:
```json
{
  "query_id": "550e8400-e29b-41d4-a716-446655440000",
  "rating": 4,
  "comments": "Good analysis but could have more depth on economic implications"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| query_id | string | Yes | The ID of the query to provide feedback for |
| rating | integer | Yes | Rating from 1-5 (1=poor, 5=excellent) |
| comments | string | No | Optional user comments on the analysis |

**Response**:
```json
{
  "success": true
}
```

### Get Evaluation Reports

Retrieve model comparison and evaluation reports.

**Endpoint**: `GET /evaluation/reports`

**Response**:
```json
{
  "model_comparison": {
    "average_ratings": {
      "gpt-3.5-turbo": 4.2,
      "gpt-4": 4.7,
      "tinyllama-1.1b": 3.8
    },
    "performance_metrics": {
      "average_latency": {
        "gpt-3.5-turbo": 2.5,
        "gpt-4": 6.8,
        "tinyllama-1.1b": 4.2
      },
      "token_efficiency": {
        "gpt-3.5-turbo": 450,
        "gpt-4": 380,
        "tinyllama-1.1b": 520
      }
    }
  },
  "task_performance": {
    "reformatting": {
      "gpt-3.5-turbo": 0.8,
      "gpt-4": 2.1,
      "tinyllama-1.1b": 2.7
    },
    "news_query_extraction": {
      "gpt-3.5-turbo": 0.7,
      "gpt-4": 1.9,
      "tinyllama-1.1b": 2.3
    },
    "summarization": {
      "gpt-3.5-turbo": 1.2,
      "gpt-4": 3.5,
      "tinyllama-1.1b": 4.2
    }
  }
}
```

## Error Responses

The API returns standard HTTP status codes:

- `200 OK`: The request succeeded
- `400 Bad Request`: The request was invalid (e.g., missing required fields)
- `500 Internal Server Error`: An error occurred on the server

Error responses include a JSON object with an `error` field:

```json
{
  "error": "Error message describing what went wrong"
}
```

## Examples

### cURL Examples

**Analyze Text**:
```bash
curl -X POST http://localhost:5000/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "text": "What are the key policy differences between Democrats and Republicans on healthcare?",
    "model": "tinyllama-1.1b"
  }'
```

**Submit Feedback**:
```bash
curl -X POST http://localhost:5000/feedback \
  -H "Content-Type: application/json" \
  -d '{
    "query_id": "550e8400-e29b-41d4-a716-446655440000",
    "rating": 4,
    "comments": "Good analysis but could use more examples"
  }'
```

### Python Examples

```python
import requests

# Analyze text
response = requests.post(
    "http://localhost:5000/analyze",
    json={
        "text": "What are the key policy differences between Democrats and Republicans on healthcare?",
        "model": "tinyllama-1.1b"
    }
)
result = response.json()
print(result["summary"])

# Submit feedback
query_id = result["query_id"]
feedback_response = requests.post(
    "http://localhost:5000/feedback",
    json={
        "query_id": query_id,
        "rating": 5,
        "comments": "Excellent balanced analysis"
    }
)
```

## Rate Limiting

There is currently no rate limiting implemented. When using local open-source models, be aware of hardware limitations that may affect performance under heavy load. 