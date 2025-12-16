"""
Embedding Generator Module
Generates text embeddings using Google's Gemini API.
"""

import os
import pickle
from pathlib import Path
from typing import List, Dict, Optional
import google.generativeai as genai
from tqdm import tqdm
from dotenv import load_dotenv


class EmbeddingGenerator:
    """
    Handles text embedding generation using Google's Gemini API.
    Supports caching to avoid redundant API calls.
    """
    
    def __init__(self, api_key: Optional[str] = None, cache_dir: str = "embeddings_cache"):
        """
        Initialize the embedding generator.
        
        Args:
            api_key: Google Gemini API key. If None, loads from environment.
            cache_dir: Directory to cache embeddings locally.
        """
        # Load environment variables
        load_dotenv()
        
        # Get API key from parameter or environment
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "GEMINI_API_KEY not found. Please set it in .env file or pass it as a parameter."
            )
        
        # Configure Gemini API
        genai.configure(api_key=self.api_key)
        
        # Set up caching
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.cache_file = self.cache_dir / "embeddings_cache.pkl"
        self.cache = self._load_cache()
        
        print(f" Embedding Generator initialized with Gemini API")
    
    def _load_cache(self) -> Dict:
        """Load cached embeddings from disk."""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, "rb") as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"Warning: Could not load cache: {e}")
                return {}
        return {}
    
    def _save_cache(self):
        """Save embeddings cache to disk."""
        try:
            with open(self.cache_file, "wb") as f:
                pickle.dump(self.cache, f)
        except Exception as e:
            print(f"Warning: Could not save cache: {e}")
    
    def generate_embedding(self, text: str, task_type: str = "retrieval_document") -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Input text to embed.
            task_type: Type of embedding task (retrieval_document, retrieval_query, etc.)
        
        Returns:
            List of floats representing the embedding vector.
        """
        if not text or not text.strip():
            raise ValueError("Empty text passed for embedding")
        # Check cache first
        cache_key = f"{task_type}:{text}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        try:
            # Generate embedding using Gemini
            result = genai.embed_content(
                model="models/text-embedding-004",
                content=text,
                task_type=task_type
            )
            embedding = result['embedding']
            
            # Cache the result
            self.cache[cache_key] = embedding
            self._save_cache()
            
            return embedding
        
        except Exception as e:
            raise RuntimeError(f"Failed to generate embedding: {e}")
    
    def generate_embeddings_batch(
        self, 
        texts: List[str], 
        task_type: str = "retrieval_document",
        show_progress: bool = True
    ) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of input texts.
            task_type: Type of embedding task.
            show_progress: Whether to show progress bar.
        
        Returns:
            List of embedding vectors.
        """
        embeddings = []
        
        iterator = tqdm(texts, desc="Generating embeddings") if show_progress else texts
        
        for text in iterator:
            embedding = self.generate_embedding(text, task_type)
            embeddings.append(embedding)
        
        return embeddings
    
    def get_embedding_dimension(self) -> int:
        """
        Get the dimensionality of embeddings from this model.
        
        Returns:
            Embedding dimension size.
        """
        # Gemini text-embedding-004 produces 768-dimensional embeddings
        test_embedding = self.generate_embedding("test", task_type="retrieval_document")
        return len(test_embedding)
    
    def clear_cache(self):
        """Clear the embeddings cache."""
        self.cache = {}
        if self.cache_file.exists():
            self.cache_file.unlink()
        print(" Cache cleared")


# Example usage
if __name__ == "__main__":
    try:
        # Initialize generator
        generator = EmbeddingGenerator()
        
        # Sample texts
        sample_texts = [
            "Artificial intelligence is transforming the world.",
            "Machine learning models require large datasets.",
            "Vector databases enable semantic search.",
        ]
        
        # Generate embeddings
        print("\nGenerating sample embeddings...")
        embeddings = generator.generate_embeddings_batch(sample_texts)
        
        # Display results
        print(f"\n Generated {len(embeddings)} embeddings")
        print(f" Embedding dimension: {len(embeddings[0])}")
        print(f" First embedding (truncated): {embeddings[0][:5]}...")
        
    except Exception as e:
        print(f"✗ Error: {e}")
