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
  "model": "tinyllama-1.1b"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| text | string | Yes | The political text to analyze |
| model | string | No | The model to use (defaults to "gpt-3.5-turbo") |

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
  "performance_metrics": {
    "total_process_time": 4.32,
    "reformatting_time": 0.78,
    "news_query_time": 0.85,
    "summarization_time": 2.69
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
```

## Rate Limiting

There is currently no rate limiting implemented. When using local open-source models, be aware of hardware limitations that may affect performance under heavy load. 