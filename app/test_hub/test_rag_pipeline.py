import asyncio
from app.utils.vector_store import VectorStore
import app.utils.llm_pipeline as llm_pipeline
from app.utils.rag_pipeline import RAGPipeline

if __name__ == "__main__":
    # Initialize dependencies
    vector_store = VectorStore()
    rag_pipeline = RAGPipeline(vector_store, llm_pipeline)

    # Example user query
    user_text = "How will the 2024 US election impact climate change policy?"

    async def run_test():
        result = await rag_pipeline.process_query(user_text)
        print("\n--- RAG Pipeline Output ---")
        print("Reformatted Query:", llm_pipeline.query_reformatter(user_text))
        print("Similar Posts:")
        for post in result['similar_posts']:
            print(post)
        print("\nNews Queries:")
        print(result['news_queries'])

    asyncio.run(run_test())
