import os
from typing import List, Dict, Any
import logging
from sentence_transformers import SentenceTransformer
import numpy as np
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VectorStore:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize the VectorStore with the specified model."""
        self.model = SentenceTransformer(model_name)
        self.embeddings = None
        self.metadata = None
        self.load_data()
    
    def load_data(self):
        """Load embeddings and metadata from disk."""
        data_dir = os.path.join(os.path.dirname(__file__), "../data/embeddings")
        
        # Load embeddings
        embeddings_path = os.path.join(data_dir, "reddit_embeddings.npy")
        if os.path.exists(embeddings_path):
            self.embeddings = np.load(embeddings_path)
        
        # Load metadata
        metadata_path = os.path.join(data_dir, "reddit_metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
    
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Perform semantic search on the vector store.
        
        Args:
            query (str): The search query
            top_k (int): Number of results to return
            
        Returns:
            List of dictionaries containing post metadata and similarity scores
        """
        if self.embeddings is None or self.metadata is None:
            raise ValueError("Vector store not initialized. Please load data first.")
        
        # Generate query embedding
        query_embedding = self.model.encode([query])[0]
        
        # Calculate similarities
        similarities = np.dot(self.embeddings, query_embedding) / (
            np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_embedding)
        )
        
        # Get top-k results
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            result = self.metadata[idx].copy()
            result['similarity_score'] = float(similarities[idx])
            results.append(result)
        
        return results
    
    def add_documents(self, documents: List[Dict]):
        """
        Add new documents to the vector store.
        
        Args:
            documents (List[Dict]): List of document dictionaries with 'text' field
        """
        # Generate embeddings for new documents
        texts = [doc['text'] for doc in documents]
        new_embeddings = self.model.encode(texts)
        
        # Update embeddings
        if self.embeddings is None:
            self.embeddings = new_embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, new_embeddings])
        
        # Update metadata
        if self.metadata is None:
            self.metadata = documents
        else:
            self.metadata.extend(documents)
        
        # Save updated data
        self.save_data()
    
    def save_data(self):
        """Save embeddings and metadata to disk."""
        data_dir = os.path.join(os.path.dirname(__file__), "../data")
        os.makedirs(data_dir, exist_ok=True)
        
        # Save embeddings
        np.save(os.path.join(data_dir, "reddit_embeddings.npy"), self.embeddings)
        
        # Save metadata
        with open(os.path.join(data_dir, "reddit_metadata.json"), 'w') as f:
            json.dump(self.metadata, f) 