import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Tuple

class VectorIndex:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize the vector index.
        
        Args:
            model_name (str): Sentence-transformers model name.
        """
        try:
            self.model = SentenceTransformer(model_name)
            self.dimension = self.model.get_sentence_embedding_dimension()
            self.index = None
        except Exception as e:
            print(f"Error initializing SentenceTransformer: {str(e)}")
            raise
    
    def build_index(self, texts: List[str]) -> None:
        """
        Build FAISS index from text descriptions.
        
        Args:
            texts (List[str]): Text descriptions to encode.
        """
        try:
            print(f"Encoding {len(texts)} texts...")
            embeddings = self.model.encode(texts, batch_size=32, show_progress_bar=True)
            embeddings = np.array(embeddings).astype('float32')
            
            # Initialize FAISS index
            self.index = faiss.IndexFlatL2(self.dimension)
            self.index.add(embeddings)
            print(f"Built FAISS index with {self.index.ntotal} vectors.")
            
        except Exception as e:
            print(f"Error building FAISS index: {str(e)}")
            raise
    
    def search(self, query: str, k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search the FAISS index.
        
        Args:
            query (str): Query text.
            k (int): Number of results.
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Distances and indices.
        """
        try:
            if self.index is None:
                raise ValueError("FAISS index not built.")
                
            query_embedding = self.model.encode([query], batch_size=1).astype('float32')
            distances, indices = self.index.search(query_embedding, k)
            return distances, indices
            
        except Exception as e:
            print(f"Error during FAISS search: {str(e)}")
            raise