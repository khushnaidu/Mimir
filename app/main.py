from flask import Flask, request, jsonify
import os
from dotenv import load_dotenv

from app.utils.vector_store import VectorStore
from app.utils.llm_pipeline import query_reformatter, news_query_extractor, context_summarizer
from app.utils.rag_pipeline import RAGPipeline
from app.utils.news_api import NewsAPIClient
import asyncio

# Load environment variables
load_dotenv()
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")

app = Flask(__name__)

# Initialize components
vector_store = VectorStore()
llm_pipeline = type('LLMPipeline', (), {
    'query_reformatter': staticmethod(query_reformatter),
    'news_query_extractor': staticmethod(news_query_extractor),
    'context_summarizer': staticmethod(context_summarizer)
})()
rag_pipeline = RAGPipeline(vector_store, llm_pipeline)
news_client = NewsAPIClient(api_key=NEWSAPI_KEY)

@app.route('/analyze', methods=['POST'])
def analyze_text():
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({'error': 'No text provided'}), 400
    text = data['text']
    print("[DEBUG] Received text:", text)
    
    try:
        async def process():
            rag_results = await rag_pipeline.process_query(text)
            print("[DEBUG] Reformatted query:", rag_results.get('reformatted_query', 'N/A'))
            #print("[DEBUG] Similar posts:", rag_results['similar_posts'])
            print("[DEBUG] News queries:", rag_results['news_queries'])
            
            # Process news queries correctly
            news_query_string = rag_results['news_queries']
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
            
            print("[DEBUG] Processed NewsAPI queries to send:", queries)
            news_results = await news_client.search_news(queries)
            print("[DEBUG] News results:", news_results)
            
            # If no results, try a fallback strategy 
            if news_results and all(res.get('totalResults', 0) == 0 for res in news_results):
                print("[DEBUG] No results found, trying alternative query strategies")
                
                # Fallback 1: Try using single keywords
                if len(queries) > 1:
                    print("[DEBUG] Trying individual keywords...")
                    # Extract all individual words from our keyword sets
                    all_words = []
                    for query in queries:
                        words = query.split()
                        all_words.extend([w for w in words if len(w) > 3])  # Only use words longer than 3 chars
                    
                    # Use top 5 longest words as they're likely more meaningful
                    all_words.sort(key=len, reverse=True)
                    single_word_queries = all_words[:5]
                    
                    if single_word_queries:
                        print("[DEBUG] Trying with single words:", single_word_queries)
                        news_results = await news_client.search_news(single_word_queries)
                        print("[DEBUG] Single word results:", news_results)
                
                # Fallback 2: If still no results, try our original method of breaking down queries
                if news_results and all(res.get('totalResults', 0) == 0 for res in news_results):
                    print("[DEBUG] Still no results, trying to further split queries")
                    refined_queries = []
                    for query in queries:
                        if ' ' in query:
                            words = query.split()
                            # If query has multiple words, try different combinations
                            if len(words) > 2:
                                refined_queries.append(' '.join(words[:2]))
                                if len(words) > 3:
                                    refined_queries.append(' '.join(words[2:4]))
                        else:
                            refined_queries.append(query)
                    
                    if refined_queries and refined_queries != queries:
                        print("[DEBUG] Final refined queries:", refined_queries)
                        news_results = await news_client.search_news(refined_queries)
                        print("[DEBUG] Final refined results:", news_results)
                
            all_articles = []
            for result in news_results:
                if 'articles' in result:
                    all_articles.extend(result['articles'])
                elif 'results' in result:
                    all_articles.extend(result['results'])
                else:
                    all_articles.append(result)

            # Create a more sophisticated relevance scoring system
            def score_article_relevance(article, original_text):
                # Guard against None article
                if article is None:
                    return 0
                    
                score = 0
                
                # Check if title contains any of our keywords
                for keyword in news_query_string.split(','):
                    keyword = keyword.strip().lower()
                    if keyword and len(keyword) > 3:
                        # Get title safely and convert to lowercase
                        title = (article.get('title') or '').lower()
                        if keyword in title:
                            score += 10  # High score for keyword in title
                        
                        # Check description and content safely
                        description = (article.get('description') or '').lower()
                        content = (article.get('content') or '').lower()
                        
                        if keyword in description:
                            score += 5  # Medium score for keyword in description
                        if keyword in content:
                            score += 2  # Lower score for keyword in content
                
                # Recency bonus (if publishedAt exists)
                if article.get('publishedAt'):
                    try:
                        # Simple check - we don't need to parse the date, just check if it's recent
                        if '2025' in article['publishedAt'] or '2024' in article['publishedAt']:
                            score += 3  # Bonus for recent articles
                    except:
                        pass
                
                return score
                
            # Sort articles by relevance score
            if all_articles:
                try:
                    all_articles.sort(key=lambda article: score_article_relevance(article, text), reverse=True)
                except Exception as e:
                    print(f"[ERROR] Error sorting articles: {e}")
                    # If sorting fails, we'll use the articles as they are
            
            # Limit the number of Reddit posts and news articles
            MAX_REDDIT_POSTS = 5
            MAX_NEWS_ARTICLES = 5

            reddit_posts = rag_results['similar_posts'][:MAX_REDDIT_POSTS]
            news_articles = all_articles[:MAX_NEWS_ARTICLES]  # Always take the top 5 after sorting

            # Truncate long text/content fields
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
                print(f"[ERROR] Error truncating Reddit posts: {e}")
                
            try:
                for article in news_articles:
                    if article is None:
                        continue
                    if 'content' in article and article['content']:
                        article['content'] = truncate(article['content'])
                    if 'description' in article and article['description']:
                        article['description'] = truncate(article['description'])
            except Exception as e:
                print(f"[ERROR] Error truncating news articles: {e}")

            context = (
                f"News Articles: {news_articles}\n"
                f"Reddit Posts: {reddit_posts}"
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
    except Exception as e:
        print(f"[ERROR] Error processing analysis request: {e}")
        return jsonify({
            'error': 'An error occurred while processing your request. Please try again with different text.',
            'summary': 'Unable to analyze the selected text due to a technical issue.',
            'raw_context': {
                'reddit_posts': [],
                'news_articles': []
            }
        }), 500

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