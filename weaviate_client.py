"""
Weaviate Client Module (v3 API - HTTP Only)
Handles schema setup, data insertion, and similarity search operations.
"""

import os
from typing import List, Dict, Any, Optional
import weaviate
from dotenv import load_dotenv
from tqdm import tqdm


class WeaviateVectorDB:
    """
    Manages Weaviate vector database operations using v3 HTTP REST API.
    """
    
    def __init__(
        self, 
        url: Optional[str] = None,
        collection_name: str = "Document"
    ):
        """
        Initialize Weaviate client connection.
        
        Args:
            url: Weaviate instance URL. If None, loads from environment.
            collection_name: Name of the collection (class) to use.
        """
        # Load environment variables
        load_dotenv()
        
        # Get Weaviate URL
        self.url = url or os.getenv("WEAVIATE_URL", "http://localhost:8081")
        self.collection_name = collection_name
        self.client = None
        
        # Connect to Weaviate
        self._connect()
    
    def _connect(self):
        """Establish connection to Weaviate instance."""
        try:
            self.client = weaviate.Client(self.url)
            
            # Test connection
            if self.client.is_ready():
                print(f"   Connected to Weaviate at {self.url}")
            else:
                raise ConnectionError("Weaviate is not ready")
                
        except Exception as e:
            raise ConnectionError(
                f"Failed to connect to Weaviate at {self.url}. "
                f"Make sure Weaviate is running. Error: {e}"
            )
    
    def create_schema(
        self, 
        vector_dimension: int,
        ef_construction: int = 128,
        ef: int = 64,
        max_connections: int = 32,
        distance_metric: str = "cosine"
    ):
        """
        Create or update the schema with HNSW index configuration.
        
        Args:
            vector_dimension: Dimensionality of embedding vectors.
            ef_construction: Size of dynamic candidate list for construction.
            ef: Size of dynamic candidate list for search.
            max_connections: Maximum number of connections per element.
            distance_metric: Distance metric (cosine, l2-squared, dot, hamming, manhattan).
        """
        try:
            # Delete existing class if it exists
            if self.client.schema.exists(self.collection_name):
                self.client.schema.delete_class(self.collection_name)
                print(f"   Deleted existing class: {self.collection_name}")
            
            # Create class schema with HNSW configuration
            class_obj = {
                "class": self.collection_name,
                "description": f"Collection for vector similarity search",
                "vectorizer": "none",  # We provide our own vectors
                "properties": [
                    {
                        "name": "text",
                        "dataType": ["text"],
                        "description": "The text content"
                    },
                    {
                        "name": "metadata",
                        "dataType": ["text"],
                        "description": "Additional metadata"
                    }
                ],
                "vectorIndexType": "hnsw",
                "vectorIndexConfig": {
                    "ef": ef,
                    "efConstruction": ef_construction,
                    "maxConnections": max_connections,
                    "distance": distance_metric,
                    "vectorCacheMaxObjects": 1000000
                }
            }
            
            self.client.schema.create_class(class_obj)
            
            print(f"   Created collection: {self.collection_name}")
            print(f"  - Vector dimension: {vector_dimension}")
            print(f"  - HNSW ef_construction: {ef_construction}")
            print(f"  - HNSW ef: {ef}")
            print(f"  - HNSW max_connections: {max_connections}")
            print(f"  - Distance metric: {distance_metric}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to create schema: {e}")
    
    def insert_data(
        self, 
        texts: List[str], 
        embeddings: List[List[float]],
        metadata: Optional[List[str]] = None,
        batch_size: int = 100,
        show_progress: bool = True
    ) -> int:
        """
        Insert texts and their embeddings into Weaviate.
        
        Args:
            texts: List of text documents.
            embeddings: List of embedding vectors.
            metadata: Optional metadata for each document.
            batch_size: Number of objects to insert per batch.
            show_progress: Whether to show progress bar.
        
        Returns:
            Number of successfully inserted objects.
        """
        if len(texts) != len(embeddings):
            raise ValueError("Number of texts and embeddings must match")
        
        if metadata is None:
            metadata = ["" for _ in texts]
        
        try:
            self.client.batch.configure(batch_size=batch_size)
            
            with self.client.batch as batch:
                iterator = tqdm(
                    zip(texts, embeddings, metadata),
                    total=len(texts),
                    desc="Inserting data"
                ) if show_progress else zip(texts, embeddings, metadata)
                
                for text, embedding, meta in iterator:
                    properties = {
                        "text": text,
                        "metadata": meta
                    }
                    
                    batch.add_data_object(
                        data_object=properties,
                        class_name=self.collection_name,
                        vector=embedding
                    )
            
            print(f"   Inserted {len(texts)} objects into Weaviate")
            return len(texts)
            
        except Exception as e:
            raise RuntimeError(f"Failed to insert data: {e}")
    
    def search(
        self, 
        query_vector: List[float], 
        limit: int = 10,
        return_metadata: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Perform similarity search using a query vector.
        
        Args:
            query_vector: Query embedding vector.
            limit: Number of results to return.
            return_metadata: Whether to include metadata in results.
        
        Returns:
            List of search results with text, distance, and metadata.
        """
        try:
            query_result = (
                self.client.query
                .get(self.collection_name, ["text", "metadata"] if return_metadata else ["text"])
                .with_near_vector({"vector": query_vector})
                .with_limit(limit)
                .with_additional(["distance"])
                .do()
            )
            
            results = []
            if "data" in query_result and "Get" in query_result["data"]:
                objects = query_result["data"]["Get"][self.collection_name]
                for obj in objects:
                    result = {
                        "text": obj.get("text", ""),
                        "distance": obj.get("_additional", {}).get("distance", 0.0),
                    }
                    if return_metadata:
                        result["metadata"] = obj.get("metadata", "")
                    results.append(result)
            
            return results
            
        except Exception as e:
            raise RuntimeError(f"Search failed: {e}")
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the current collection.
        
        Returns:
            Dictionary with collection statistics.
        """
        try:
            aggregate_result = (
                self.client.query
                .aggregate(self.collection_name)
                .with_meta_count()
                .do()
            )
            
            count = 0
            if "data" in aggregate_result and "Aggregate" in aggregate_result["data"]:
                agg_data = aggregate_result["data"]["Aggregate"][self.collection_name]
                if agg_data and len(agg_data) > 0:
                    count = agg_data[0].get("meta", {}).get("count", 0)
            
            return {
                "collection_name": self.collection_name,
                "total_objects": count,
            }
        except Exception as e:
            return {"error": str(e)}
    
    def delete_collection(self):
        """Delete the current collection."""
        try:
            if self.client.schema.exists(self.collection_name):
                self.client.schema.delete_class(self.collection_name)
                print(f"   Deleted collection: {self.collection_name}")
        except Exception as e:
            print(f"Warning: Failed to delete collection: {e}")
    
    def close(self):
        """Close the Weaviate client connection (no-op for v3)."""
        # V3 client doesn't need explicit close
        pass


# Example usage
if __name__ == "__main__":
    try:
        # Initialize client
        db = WeaviateVectorDB()
        
        # Create schema
        db.create_schema(vector_dimension=768, ef_construction=128, ef=64)
        
        # Sample data
        sample_texts = [
            "Vector databases are optimized for similarity search.",
            "HNSW is an efficient approximate nearest neighbor algorithm.",
            "Embeddings capture semantic meaning of text."
        ]
        
        # Generate dummy embeddings (in real use, use EmbeddingGenerator)
        import random
        sample_embeddings = [[random.random() for _ in range(768)] for _ in range(3)]
        
        # Insert data
        db.insert_data(sample_texts, sample_embeddings)
        
        # Get stats
        stats = db.get_collection_stats()
        print(f"\nCollection stats: {stats}")
        
        # Perform search
        query_vector = [random.random() for _ in range(768)]
        results = db.search(query_vector, limit=3)
        print(f"\nSearch returned {len(results)} results")
        
        # Close connection
        db.close()
        
    except Exception as e:
        print(f"✗ Error: {e}")
