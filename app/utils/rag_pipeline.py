from app.utils.vector_store import VectorStore
from app.utils.llm_pipeline import query_reformatter, news_query_extractor

class RAGPipeline:
    def __init__(self, vector_store, llm_pipeline):
        self.vector_store = vector_store
        self.llm_pipeline = llm_pipeline

    async def process_query(self, user_text):
        # Step 1: Reformat the query for semantic search
        reformatted_query = self.llm_pipeline.query_reformatter(user_text)
        # Step 2: Search vector store for similar posts
        similar_posts = self.vector_store.search(reformatted_query)
        # Step 3: Extract news search queries
        news_queries = self.llm_pipeline.news_query_extractor(user_text)
        return {
            'similar_posts': similar_posts,
            'news_queries': news_queries
        }
