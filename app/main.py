from flask import Flask, request, jsonify
from utils.query_processor import QueryProcessor
from utils.vector_store import VectorStore
from utils.data_acquisition import RedditDataFetcher
from newsapi import NewsApiClient
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Initialize components
query_processor = QueryProcessor()
vector_store = VectorStore()
reddit_fetcher = RedditDataFetcher()
news_api = NewsApiClient(api_key=os.getenv('NEWS_API_KEY'))

@app.route('/analyze', methods=['POST'])
def analyze_text():
    """
    Analyze highlighted text and return relevant context.
    
    Expected JSON payload:
    {
        "text": "Text to analyze"
    }
    """
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({'error': 'No text provided'}), 400
    
    text = data['text']
    
    # Process query
    analysis = query_processor.analyze_text(text)
    
    # Get Reddit results
    reddit_results = vector_store.search(analysis['reddit_query'])
    
    # Get news results
    news_components = analysis['news_api_components']
    news_results = news_api.get_everything(
        q=news_components['search_terms'],
        language='en',
        sort_by='relevancy',
        page_size=5
    )
    
    # Combine and analyze results
    context = {
        'reddit_posts': reddit_results,
        'news_articles': news_results['articles'],
        'analysis': analysis
    }
    
    return jsonify(context)

@app.route('/update_data', methods=['POST'])
def update_data():
    """Update the vector store with new Reddit data."""
    try:
        # Fetch recent posts
        posts = reddit_fetcher.fetch_recent_posts()
        
        # Add to vector store
        vector_store.add_documents(posts)
        
        return jsonify({'message': f'Successfully updated {len(posts)} posts'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True) 