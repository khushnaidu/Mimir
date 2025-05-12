from flask import Flask, request, jsonify
import os

from app.utils.vector_store import VectorStore
from app.utils.llm_pipeline import query_reformatter, news_query_extractor, context_summarizer
from app.utils.rag_pipeline import RAGPipeline
from app.utils.news_api import NewsAPIClient
import asyncio

# Load environment variables


app = Flask(__name__)

# Initialize components
vector_store = VectorStore()
llm_pipeline = type('LLMPipeline', (), {
    'query_reformatter': staticmethod(query_reformatter),
    'news_query_extractor': staticmethod(news_query_extractor),
    'context_summarizer': staticmethod(context_summarizer)
})()
rag_pipeline = RAGPipeline(vector_store, llm_pipeline)
news_client = NewsAPIClient(api_key="5f127c1cd18842fd8380c946fd50ad63")

@app.route('/analyze', methods=['POST'])
def analyze_text():
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({'error': 'No text provided'}), 400
    text = data['text']
    print("[DEBUG] Received text:", text)
    async def process():
        rag_results = await rag_pipeline.process_query(text)
        print("[DEBUG] Reformatted query:", rag_results.get('reformatted_query', 'N/A'))
        #print("[DEBUG] Similar posts:", rag_results['similar_posts'])
        print("[DEBUG] News queries:", rag_results['news_queries'])
        queries = rag_results['news_queries']
        if isinstance(queries, str):
            queries = [queries]
        print("[DEBUG] NewsAPI queries to send:", queries)
        news_results = await news_client.search_news(queries)
        print("[DEBUG] News results:", news_results)
        all_articles = []
        for result in news_results:
            if 'articles' in result:
                all_articles.extend(result['articles'])
            elif 'results' in result:
                all_articles.extend(result['results'])
            else:
                all_articles.append(result)

        # Limit the number of Reddit posts and news articles
        MAX_REDDIT_POSTS = 5
        MAX_NEWS_ARTICLES = 5

        reddit_posts = rag_results['similar_posts'][:MAX_REDDIT_POSTS]
        news_articles = all_articles[:MAX_NEWS_ARTICLES]

        # Truncate long text/content fields
        def truncate(text, max_chars=500):
            return text[:max_chars] + ('...' if len(text) > max_chars else '')

        for post in reddit_posts:
            if 'text' in post and post['text']:
                post['text'] = truncate(post['text'])
            if 'top_comments' in post:
                for comment in post['top_comments']:
                    if 'body' in comment and comment['body']:
                        comment['body'] = truncate(comment['body'])
        for article in news_articles:
            if 'content' in article and article['content']:
                article['content'] = truncate(article['content'])
            if 'description' in article and article['description']:
                article['description'] = truncate(article['description'])

        context = (
            f"Reddit Posts: {reddit_posts}\n"
            f"News Articles: {news_articles}"
        )
        print("[DEBUG] Final news_articles sent to frontend:", news_articles)
        print("[DEBUG] Context sent to LLM:", context)
        summary = llm_pipeline.context_summarizer(context)
        print("[DEBUG] LLM Summary:", summary)
        return {
            'summary': summary,
            'raw_context': {
                'reddit_posts': rag_results['similar_posts'],
                'news_articles': news_articles
            }
        }
    result = asyncio.run(process())
    return jsonify(result)

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