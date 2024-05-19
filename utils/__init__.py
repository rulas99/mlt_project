from sentence_transformers import SentenceTransformer, quantization
import numpy as np
from typing import Dict

def hit_rate_at_k(queries: Dict[str, str], corpus: Dict[str, str], relevant_docs: Dict[str, str], 
                  model: SentenceTransformer, k: int = 10, quantization_int8:bool=False) -> float:
    """
    Calculate the hit rate metric for an embedding model.

    Args:
        queries (Dict[str, str]): Dictionary of queries in the format {qid: query}.
        corpus (Dict[str, str]): Dictionary of documents in the format {cid: doc}.
        relevant_docs (Dict[str, str]): Dictionary mapping queries to relevant documents {qid: cid}.
        model (SentenceTransformer): Sentence Transformers model to obtain embeddings.
        k (int, optional): Number of top similar documents to consider for calculating hit rate. 
                           Defaults to 10.
        quantization_int8 (bool, optional): Whether to apply int8 quantization to embeddings.
                           Defaults to False.

    Returns:
        float: The hit rate metric indicating the proportion of queries for which the relevant document 
               is among the top k most similar ones.
    """
    
    # Obtain embeddings for the queries and corpus
    query_embeddings = model.encode(list(queries.values()))
    corpus_embeddings = model.encode(list(corpus.values()))
    
    if quantization_int8:
        query_embeddings = quantization.quantize_embeddings(query_embeddings, precision="int8")
        corpus_embeddings = quantization.quantize_embeddings(corpus_embeddings, precision="int8")
    
    # Calculate similarities
    dot_score = np.dot(query_embeddings, corpus_embeddings.T)
    
    # Initialize hit counter
    hits = 0
    
    # Convert corpus keys to an array for quick access
    corpus_keys_array = np.array(list(corpus.keys()))
    
    # For each query
    for idx, qid in enumerate(queries.keys()):
        # Get indices of documents sorted by descending similarity
        top_k_indices = np.argsort(dot_score[idx])[::-1][:k]
        
        # Check if the relevant document is among the top k most similar documents
        relevant_cid = relevant_docs[qid]
        if relevant_cid in corpus_keys_array[top_k_indices]:
            hits += 1
    
    # Calculate the hit rate
    hit_rate = hits / len(queries)
    
    return hit_rate
