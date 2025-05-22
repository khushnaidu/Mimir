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
from app.utils.political_analysis import PoliticalAnalyzer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load env vars
load_dotenv()
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")

app = Flask(__name__)
CORS(app)

# Init components
vector_store = VectorStore()
news_client = NewsAPIClient(api_key=NEWSAPI_KEY)

from app.utils.llm_pipeline import query_reformatter, news_query_extractor, context_summarizer, call_model

# Init PoliticalAnalyzer
political_analyzer = PoliticalAnalyzer(model_name="tinyllama-1.1b")

class CustomLLMPipeline:
    def __init__(self):
        self.political_analyzer = political_analyzer
    
    def query_reformatter(self, text, model_name):
        if model_name == "tinyllama-1.1b":
            messages = [
                {"role": "system", "content": "You are a political discourse expert who specializes in detecting key entities, policies, events and contextual relationships in political text."},
                {"role": "user", "content": f"Reformat the following text to be optimal for semantic search in a Reddit political discussion context: {text}"}
            ]
            return call_model("tinyllama-1.1b", messages, task_type="query_reformatter")
        else:
            return query_reformatter(text, model_name)
    
    def news_query_extractor(self, text, model_name):
        if model_name == "tinyllama-1.1b":
            return self.political_analyzer.extract_policy_keywords(text)
        else:
            return news_query_extractor(text, model_name)
    
    def context_summarizer(self, context, model_name):
        if model_name == "tinyllama-1.1b":
            messages = [
                {"role": "system", "content": "You are a balanced political analyst who provides nuanced perspectives across the political spectrum."},
                {"role": "user", "content": f"Provide a comprehensive summary of the following context, highlighting different perspectives and key insights:\n\n{context}"}
            ]
            return call_model("tinyllama-1.1b", messages, task_type="context_summarizer")
        else:
            return context_summarizer(context, model_name)

llm_pipeline = CustomLLMPipeline()
rag_pipeline = RAGPipeline(vector_store, llm_pipeline)

@app.route('/models', methods=['GET'])
def get_available_models():
    models = list(AVAILABLE_MODELS.keys())
    
    model_info = {
        'models': models,
        'default_model': 'gpt-3.5-turbo',
        'model_descriptions': {
            'gpt-3.5-turbo': 'OpenAI GPT-3.5 Turbo - Fast and efficient commercial model',
            'gpt-4': 'OpenAI GPT-4 - Advanced commercial model with strong reasoning',
            'tinyllama-1.1b': 'TinyLlama with LoRA - Open-source model optimized for political analysis'
        }
    }
    
    return jsonify(model_info), 200, {'Content-Type': 'application/json'}

@app.route('/analyze', methods=['POST'])
def analyze_text():
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({'error': 'No text provided'}), 400
    
    text = data['text']
    model_name = data.get('model', 'gpt-3.5-turbo')
    collect_feedback = data.get('collect_feedback', True)
    
    logger.info(f"Received text: {text[:50]}...")
    logger.info(f"Using model: {model_name}")
    
    try:
        async def process():
            query_id = str(uuid.uuid4())
            process_start_time = time.time()
            
            # Step 1: Reformat query
            reformatted_query_result = llm_pipeline.query_reformatter(text, model_name)
            reformatted_query = reformatted_query_result.content
            logger.info(f"Reformatted query: {reformatted_query}")
            
            # Step 2: Get similar posts
            similar_posts = await rag_pipeline.search_vector_store(reformatted_query)
            
            # Step 3: Extract news queries
            news_query_result = llm_pipeline.news_query_extractor(text, model_name)
            news_query_string = news_query_result.content
            logger.info(f"News queries: {news_query_string}")
            
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
            else:
                queries = [news_query_string]
            
            # Limit queries
            queries = queries[:4]
            
            logger.info(f"Processed NewsAPI queries: {queries}")
            news_results = await news_client.search_news(queries)
            
            # Collect articles
            all_articles = []
            for result in news_results:
                if 'articles' in result:
                    all_articles.extend(result['articles'])
                elif 'results' in result:
                    all_articles.extend(result['results'])
                else:
                    all_articles.append(result)
            
            # Limit the number of results
            MAX_REDDIT_POSTS = 5
            MAX_NEWS_ARTICLES = 5
            reddit_posts = similar_posts[:MAX_REDDIT_POSTS]
            news_articles = all_articles[:MAX_NEWS_ARTICLES]
            
            # Truncate long text fields
            def truncate(text, max_chars=500):
                if text is None:
                    return ""
                return text[:max_chars] + ('...' if len(text) > max_chars else '')
            
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
            
            # Create context for summarization
            context = (
                f"News Articles: {news_articles}\n"
                f"Reddit Posts: {reddit_posts}"
            )
            
            # Generate summary
            summary_result = llm_pipeline.context_summarizer(context, model_name)
            summary = summary_result.content
            
            process_end_time = time.time()
            total_process_time = process_end_time - process_start_time
            
            logger.info(f"Summary generated in {total_process_time:.2f}s")
            
            # Collect evaluation data
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
                    "user_feedback": None
                }
                
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
    data = request.get_json()
    if not data or 'query_id' not in data or 'rating' not in data:
        return jsonify({'error': 'Missing required fields'}), 400
    
    query_id = data['query_id']
    rating = data['rating']
    comments = data.get('comments', '')
    
    try:
        update_evaluation_feedback(query_id, rating, comments)
        return jsonify({'success': True})
    except Exception as e:
        logger.error(f"Error saving feedback: {e}")
        return jsonify({'error': 'Failed to save feedback'}), 500

@app.route('/evaluation/reports', methods=['GET'])
def get_evaluation_reports():
    try:
        reports = generate_model_comparison_reports()
        return jsonify(reports)
    except Exception as e:
        logger.error(f"Error generating reports: {e}")
        return jsonify({'error': 'Failed to generate reports'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0') 