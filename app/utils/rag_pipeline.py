from app.utils.vector_store import VectorStore
from app.utils.llm_pipeline import query_reformatter, news_query_extractor
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGPipeline:
    def __init__(self, vector_store, llm_pipeline):
        self.vector_store = vector_store
        self.llm_pipeline = llm_pipeline
        logger.info("RAG Pipeline initialized")

    async def process_query(self, user_text):
        logger.info(f"Processing query: {user_text[:50]}...")
        
        # Step 1: Reformat the query for semantic search
        try:
            reformatted_query = self.llm_pipeline.query_reformatter(user_text)
            logger.info(f"Reformatted query: {reformatted_query}")
        except Exception as e:
            logger.error(f"Error reformatting query: {str(e)}")
            reformatted_query = user_text  # Fallback to original text
            
        # Step 2: Search vector store for similar posts
        similar_posts = await self.search_vector_store(reformatted_query)
            
        # Step 3: Extract news search queries
        try:
            news_queries = self.llm_pipeline.news_query_extractor(user_text)
            logger.info(f"Extracted news queries: {news_queries}")
        except Exception as e:
            logger.error(f"Error extracting news queries: {str(e)}")
            news_queries = user_text  # Fallback to original text
            
        return {
            'reformatted_query': reformatted_query,
            'similar_posts': similar_posts,
            'news_queries': news_queries
        }
        
    async def search_vector_store(self, reformatted_query):
        """Search vector store for similar posts"""
        try:
            # Check if vector store has data
            if not hasattr(self.vector_store, 'embeddings') or self.vector_store.embeddings is None:
                logger.warning("Vector store has no embeddings. Loading data...")
                self.vector_store.load_data()
                
            if not hasattr(self.vector_store, 'embeddings') or self.vector_store.embeddings is None:
                logger.error("Vector store failed to load embeddings")
                return []
            else:
                similar_posts = self.vector_store.search(reformatted_query)
                logger.info(f"Found {len(similar_posts)} similar posts")
                return similar_posts
        except Exception as e:
            logger.error(f"Error searching vector store: {str(e)}")
            return []
