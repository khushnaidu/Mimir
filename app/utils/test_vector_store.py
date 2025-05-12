from vector_store import VectorStore

if __name__ == "__main__":
    # Instantiate the vector store
    store = VectorStore()
    print("Loaded embeddings shape:", None if store.embeddings is None else store.embeddings.shape)
    print("Loaded metadata count:", None if store.metadata is None else len(store.metadata))

    # Test search
    test_query = "climate change policy"
    try:
        results = store.search(test_query, top_k=3)
        print(f"Search results for query: '{test_query}'")
        for i, res in enumerate(results):
            print(f"Result {i+1}: {res}")
    except Exception as e:
        print("Search failed:", e)

    # Test add_documents (if implemented)
    if hasattr(store, 'add_documents'):
        try:
            store.add_documents([{"text": "Test document for addition"}])
            print("add_documents ran without error.")
        except Exception as e:
            print("add_documents failed:", e)
