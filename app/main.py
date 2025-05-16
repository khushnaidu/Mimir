from flask import Flask, request, jsonify
import os
from dotenv import load_dotenv
import asyncio
import time
import json
import uuid
from datetime import datetime
from flask_cors import CORS
import logging

from app.utils.vector_store import VectorStore
from app.utils.llm_pipeline import AVAILABLE_MODELS
from app.utils.rag_pipeline import RAGPipeline
from app.utils.news_api import NewsAPIClient
from app.utils.evaluation import store_evaluation_data, update_evaluation_feedback, generate_model_comparison_reports

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize components
vector_store = VectorStore()
news_client = NewsAPIClient(api_key=NEWSAPI_KEY)

# Import and setup LLM pipeline dynamically to handle specialized class
from app.utils.llm_pipeline import query_reformatter, news_query_extractor, context_summarizer
llm_pipeline = type('LLMPipeline', (), {
    'query_reformatter': staticmethod(query_reformatter),
    'news_query_extractor': staticmethod(news_query_extractor),
    'context_summarizer': staticmethod(context_summarizer)
})()

rag_pipeline = RAGPipeline(vector_store, llm_pipeline)

@app.route('/models', methods=['GET'])
def get_available_models():
    """Return a list of available models for frontend selection"""
    return jsonify({
        'models': list(AVAILABLE_MODELS.keys()),
        'default_model': 'gpt-3.5-turbo'  # Default model
    })

@app.route('/analyze', methods=['POST'])
def analyze_text():
    """Analyze text with the selected model and capture performance metrics"""
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({'error': 'No text provided'}), 400
    
    text = data['text']
    
    # Get model selection if provided, otherwise use default
    model_name = data.get('model', 'gpt-3.5-turbo')
    collect_feedback = data.get('collect_feedback', True)  # Default to collecting feedback
    
    logger.info(f"Received text: {text[:50]}...")
    logger.info(f"Using model: {model_name}")
    
    try:
        async def process():
            # Generate a unique query ID
            query_id = str(uuid.uuid4())
            
            # Record start time for full process
            process_start_time = time.time()
            
            # Process with the selected model
            reformatted_query_result = llm_pipeline.query_reformatter(text, model_name)
            reformatted_query = reformatted_query_result.content
            
            logger.info(f"Reformatted query: {reformatted_query}")
            
            # Get similar posts from vector store
            similar_posts = await rag_pipeline.search_vector_store(reformatted_query)
            
            # Extract news queries with the selected model
            logger.info(f"Sending to news_query_extractor: {text[:50]}...")
            news_query_result = llm_pipeline.news_query_extractor(text, model_name)
            news_query_string = news_query_result.content
            
            logger.info(f"News queries: {news_query_string}")
            
            # Process news queries
            if isinstance(news_query_string, str):
                # Split the comma-separated keywords and create individual queries
                raw_keywords = [k.strip() for k in news_query_string.split(',')]
                
                # Filter out any empty keywords
                keywords = [k for k in raw_keywords if k]
                
                # Create meaningful search queries from the keywords
                # Prioritize full phrases for political context
                queries = []
                
                # First, add any full politically relevant phrases (multi-word)
                for keyword in keywords:
                    if len(keyword.split()) > 1:
                        queries.append(keyword)
                
                # If we have less than 3 queries, add important single terms
                if len(queries) < 3:
                    # Add important single terms, preferring political entities
                    political_entities = [k for k in keywords if len(k.split()) == 1 and len(k) > 3 and k.lower() not in ['the', 'and', 'with', 'from', 'that', 'this']]
                    queries.extend(political_entities[:3-len(queries)])
                    
                # Ensure we have at least one query
                if not queries and keywords:
                    queries = keywords[:3]  # Limit to top 3 keywords if nothing else
                    
                # Add one combined query with main terms for context
                if len(keywords) >= 2:
                    main_terms = ' '.join(keywords[:3])  # Combine first 3 terms
                    if main_terms not in queries:
                        queries.append(main_terms)
            else:
                queries = [news_query_string]
            
            # Cap the number of queries to prevent too broad a search
            queries = queries[:4]  # Limit to 4 queries maximum
            
            logger.info(f"Processed NewsAPI queries to send: {queries}")
            news_results = await news_client.search_news(queries)
            
            # Implement fallbacks for no results if needed
            # (Simplified for this implementation)
            
            # Collect all articles
            all_articles = []
            for result in news_results:
                if 'articles' in result:
                    all_articles.extend(result['articles'])
                elif 'results' in result:
                    all_articles.extend(result['results'])
                else:
                    all_articles.append(result)
            
            # Score and sort articles (simplified)
            # For a more sophisticated implementation, re-implement the scoring logic
            
            # Limit the number of Reddit posts and news articles
            MAX_REDDIT_POSTS = 5
            MAX_NEWS_ARTICLES = 5

            reddit_posts = similar_posts[:MAX_REDDIT_POSTS]
            news_articles = all_articles[:MAX_NEWS_ARTICLES]
            
            # Truncate long text fields
            def truncate(text, max_chars=500):
                if text is None:
                    return ""
                return text[:max_chars] + ('...' if len(text) > max_chars else '')
            
            # Truncate Reddit posts
            try:
                for post in reddit_posts:
                    if post is None:
                        continue
                    if 'text' in post and post['text']:
                        post['text'] = truncate(post['text'])
                    if 'top_comments' in post:
                        for comment in post['top_comments']:
                            if comment and 'body' in comment and comment['body']:
                                comment['body'] = truncate(comment['body'])
            except Exception as e:
                logger.error(f"Error truncating Reddit posts: {str(e)}")
            
            # Truncate news articles
            try:
                for article in news_articles:
                    if article is None:
                        continue
                    if 'content' in article and article['content']:
                        article['content'] = truncate(article['content'])
                    if 'description' in article and article['description']:
                        article['description'] = truncate(article['description'])
            except Exception as e:
                logger.error(f"Error truncating news articles: {str(e)}")
            
            # Create context for LLM summarization
            context = (
                f"News Articles: {news_articles}\n"
                f"Reddit Posts: {reddit_posts}"
            )
            
            # Generate summary with the selected model
            summary_result = llm_pipeline.context_summarizer(context, model_name)
            summary = summary_result.content
            
            # Record end time and calculate total processing time
            process_end_time = time.time()
            total_process_time = process_end_time - process_start_time
            
            logger.info(f"Summary generated in {total_process_time:.2f}s")
            
            # Collect evaluation data if requested
            if collect_feedback:
                eval_data = {
                    "query_id": query_id,
                    "timestamp": datetime.now().isoformat(),
                    "model": model_name,
                    "input_text": text,
                    "total_process_time": total_process_time,
                    "metrics": {
                        "reformatting": {
                            "latency": reformatted_query_result.latency,
                            "token_count": reformatted_query_result.token_count,
                            "result": reformatted_query
                        },
                        "news_query_extraction": {
                            "latency": news_query_result.latency,
                            "token_count": news_query_result.token_count,
                            "result": news_query_string
                        },
                        "summarization": {
                            "latency": summary_result.latency,
                            "token_count": summary_result.token_count,
                            "result": summary
                        }
                    },
                    "user_feedback": None  # To be filled by frontend later
                }
                
                # Store evaluation data for future analysis
                store_evaluation_data(eval_data)
            
            return {
                'summary': summary,
                'raw_context': {
                    'reddit_posts': reddit_posts,
                    'news_articles': news_articles
                },
                'model_used': model_name,
                'query_id': query_id if collect_feedback else None,
                'performance_metrics': {
                    'total_process_time': total_process_time,
                    'reformatting_time': reformatted_query_result.latency,
                    'news_query_time': news_query_result.latency,
                    'summarization_time': summary_result.latency
                }
            }
        
        result = asyncio.run(process())
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error processing analysis request: {e}")
        return jsonify({
            'error': 'An error occurred while processing your request. Please try again with different text.',
            'summary': 'Unable to analyze the selected text due to a technical issue.',
            'raw_context': {
                'reddit_posts': [],
                'news_articles': []
            }
        }), 500

@app.route('/feedback', methods=['POST'])
def submit_feedback():
    """Submit user feedback for a query"""
    data = request.get_json()
    if not data or 'query_id' not in data or 'rating' not in data:
        return jsonify({'error': 'Missing required fields'}), 400
    
    query_id = data['query_id']
    rating = data['rating']
    comments = data.get('comments', '')
    
    try:
        # Update the evaluation data with user feedback
        update_evaluation_feedback(query_id, rating, comments)
        return jsonify({'success': True})
    except Exception as e:
        logger.error(f"Error saving feedback: {e}")
        return jsonify({'error': 'Failed to save feedback'}), 500

@app.route('/evaluation/reports', methods=['GET'])
def get_evaluation_reports():
    """Generate and return model comparison reports"""
    try:
        reports = generate_model_comparison_reports()
        return jsonify(reports)
    except Exception as e:
        logger.error(f"Error generating reports: {e}")
        return jsonify({'error': 'Failed to generate reports'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0') 