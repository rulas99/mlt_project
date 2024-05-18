from sentence_transformers import SentenceTransformer
import numpy as np
from typing import Dict

def hit_rate_at_k(queries: Dict[str, str], corpus: Dict[str, str], relevant_docs: Dict[str, str], 
                  model: SentenceTransformer, k: int = 10) -> float:
    """
    Calcula la métrica de hit rate para un modelo de embeddings.

    Args:
        queries (Dict[str, str]): Diccionario de consultas con el formato {qid: query}.
        corpus (Dict[str, str]): Diccionario de documentos con el formato {cid: doc}.
        relevant_docs (Dict[str, str]): Diccionario que mapea consultas a documentos relevantes {qid: cid}.
        model (SentenceTransformer): Modelo de Sentence Transformers para obtener los embeddings.
        k (int, optional): Número de documentos más similares a considerar para calcular el hit rate. 
                           Por defecto es 10.

    Returns:
        float: La métrica de hit rate (tasa de aciertos) que indica la proporción de consultas 
               para las cuales el documento relevante está entre los top k más similares.
    """
    
    # Obtener los embeddings de las consultas y el corpus
    query_embeddings = model.encode(list(queries.values()))
    corpus_embeddings = model.encode(list(corpus.values()))
    
    # Calcular las similitudes
    dot_score = np.dot(query_embeddings, corpus_embeddings.T)
    
    # Inicializar el contador de aciertos
    hits = 0
    
    # Convertir las claves del corpus en un array para acceso rápido
    corpus_keys_array = np.array(list(corpus.keys()))
    
    # Para cada consulta
    for idx, qid in enumerate(queries.keys()):
        # Obtener los índices de los documentos ordenados por similitud descendente
        top_k_indices = np.argsort(dot_score[idx])[::-1][:k]
        
        # Verificar si el documento relevante está en los `top k` documentos más similares
        relevant_cid = relevant_docs[qid]
        if relevant_cid in corpus_keys_array[top_k_indices]:
            hits += 1
    
    # Calcular la tasa de aciertos
    hit_rate = hits / len(queries)
    
    return hit_rate