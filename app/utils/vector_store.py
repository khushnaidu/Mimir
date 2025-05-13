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
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", use_rich_data: bool = True):
        """Initialize the VectorStore with the specified model."""
        self.model = SentenceTransformer(model_name)
        self.embeddings = None
        self.metadata = None
        self.use_rich_data = use_rich_data
        self.load_data()

    def load_data(self):
        """Load embeddings and metadata from disk."""
        data_dir = os.path.join(os.path.dirname(__file__), "../data")

        # Load embeddings
        embeddings_path = os.path.join(data_dir, "reddit_embeddings.npy")
        if os.path.exists(embeddings_path):
            logger.info(f"Loading embeddings from {embeddings_path}")
            self.embeddings = np.load(embeddings_path)
            logger.info(f"Loaded {self.embeddings.shape[0]} embeddings with dimension {self.embeddings.shape[1]}")
        else:
            logger.warning(f"Embeddings file not found at {embeddings_path}")
            return

        # Try to load rich post data first if enabled
        if self.use_rich_data:
            rich_posts_path = os.path.join(data_dir, "reddit_posts_with_embeddings.json")
            if os.path.exists(rich_posts_path):
                try:
                    logger.info(f"Loading rich post data from {rich_posts_path}")
                    with open(rich_posts_path, 'r') as f:
                        posts_data = json.load(f)
                    
                    # Verify that we have the appropriate structure
                    if isinstance(posts_data, list) and len(posts_data) > 0 and 'embedding_index' in posts_data[0]:
                        # Sort posts by embedding_index if needed
                        posts_data.sort(key=lambda x: x['embedding_index'])
                        
                        # Validate that embedding indices match our array size
                        if len(posts_data) <= self.embeddings.shape[0]:
                            self.metadata = posts_data
                            logger.info(f"Successfully loaded {len(self.metadata)} rich post metadata entries")
                            return
                        else:
                            logger.warning(f"Mismatch between post count ({len(posts_data)}) and embedding count ({self.embeddings.shape[0]})")
                    else:
                        logger.warning("Rich post data does not have the expected structure with 'embedding_index'")
                except Exception as e:
                    logger.error(f"Error loading rich post data: {str(e)}")
        
        # Fall back to original metadata if rich data failed or is disabled
        metadata_path = os.path.join(data_dir, "reddit_metadata.json")
        if os.path.exists(metadata_path):
            logger.info(f"Loading standard metadata from {metadata_path}")
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
            logger.info(f"Loaded {len(self.metadata)} standard metadata entries")
        else:
            logger.warning(f"Metadata file not found at {metadata_path}")

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
            raise ValueError(
                "Vector store not initialized. Please load data first.")

        # Generate query embedding
        query_embedding = self.model.encode([query])[0]

        # Calculate similarities
        similarities = np.dot(self.embeddings, query_embedding) / (
            np.linalg.norm(self.embeddings, axis=1) *
            np.linalg.norm(query_embedding)
        )

        # Get top-k results
        top_indices = np.argsort(similarities)[-top_k:][::-1]

        results = []
        for idx in top_indices:
            # Handle case where idx might be out of range for metadata
            if idx < len(self.metadata):
                result = self.metadata[idx].copy()
                result['similarity_score'] = float(similarities[idx])
                results.append(result)
            else:
                logger.warning(f"Index {idx} out of range for metadata with length {len(self.metadata)}")

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

        # Save metadata - use original format for backward compatibility
        with open(os.path.join(data_dir, "reddit_metadata.json"), 'w') as f:
            json.dump(self.metadata, f)
        
        # If using rich data, also save in rich format
        if self.use_rich_data:
            # Add embedding_index to each document if not present
            for i, doc in enumerate(self.metadata):
                if 'embedding_index' not in doc:
                    doc['embedding_index'] = i
            
            with open(os.path.join(data_dir, "reddit_posts_with_embeddings.json"), 'w') as f:
                json.dump(self.metadata, f)
