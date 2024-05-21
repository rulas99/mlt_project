from sentence_transformers import SentenceTransformer, quantization
import numpy as np
from typing import Dict
import pandas as pd

from dotenv import load_dotenv
from os import environ as os_environ

from json import loads as json_loads

from groq import Groq

import telebot

from . import agent_roles


load_dotenv()

BOT_TOKEN = os_environ.get('BOT_TOKEN')
GROQ_API_KEY = os_environ.get("GROQ_API_KEY")

BOT = telebot.TeleBot(BOT_TOKEN)
CLIENT = Groq(api_key=GROQ_API_KEY)


CUSTOM_MODEL = SentenceTransformer("./results/domain_adaptation_model", device='cuda')
BASE_MODEL = SentenceTransformer(
    'sentence-transformers/all-MiniLM-L12-v2', device='cuda')

DATA = pd.read_parquet('./data/mlt_data_publications.parquet',
                       columns=['paperId', 'title', 'abstract', 'venue',
                                's2FieldsOfStudy', 'publicationDate',
                                'authors'])
TRIPLES = pd.read_parquet('./data/triples_corpus.parquet')


ENTITIES = pd.read_pickle('./data/vector_store/entities.pkl')
EMB_ENTS = pd.read_parquet('./data/vector_store/emb_ents.parquet').values

SUBSTR2REMOVE = ["'source': 's2-fos-model'",
                 "array([{'affiliations': array([], dtype=object)",
                 "'citationCount': None",
                 "'externalIds': None",
                 "'paperCount': None",
                 "'citationCount': None",
                 "'source': 'external'",
                 " , ",]


def hit_rate_at_k(queries: Dict[str, str], corpus: Dict[str, str], relevant_docs: Dict[str, str],
                  model: SentenceTransformer, k: int = 10, quantization_int8: bool = False) -> float:
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
        query_embeddings = quantization.quantize_embeddings(
            query_embeddings, precision="int8")
        corpus_embeddings = quantization.quantize_embeddings(
            corpus_embeddings, precision="int8")

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


def get_top_k_relevant_info(emb_q: np.array, k: int) -> str:
    """_summary_

    Args:
        emb_q (np.array): _description_
        k (int): _description_

    Returns:
        str: _description_
    """
    scores = np.dot(EMB_ENTS, emb_q)
    top_k = np.argsort(scores)[::-1][:k]

    top_k_entities = [(list(ENTITIES.keys())[
                       i], ENTITIES[list(ENTITIES.keys())[i]]) for i in top_k]

    ids = [i[0] for i in top_k_entities]
    
    print("**** retrieved ids ****\n", ids)

    triples = str(TRIPLES[TRIPLES.subjectId.isin(ids) |
                  TRIPLES.objectId.isin(ids)].to_dict(orient='records'))
    raw_info = str(DATA[DATA.paperId.isin(ids)].to_dict(orient='records'))

    ri = f"Top {k} relevant triples retrieved for the query:\n{triples}\n\nDetailed information on the papers referenced in the triples:\n{raw_info}"

    for substr in SUBSTR2REMOVE:
        ri = ri.replace(substr, "")

    return ri


def request_agent(user_request, role, temperature=0.2,
                  top_p=0.3, model="mixtral-8x7b-32768",
                  max_tokens=13000, response_format="json_object"):

    try:
        chat_completion = CLIENT.chat.completions.create(
            messages=[
                {
                    "content": f"{agent_roles.roles[role]}",
                    "role": "system"
                },
                {
                    "role": "user",
                    "content": f"{user_request}",
                }
            ],
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            model=model,
            timeout=30,
            response_format={"type": response_format} if response_format=="json_object" else None,
        )
        
        response = chat_completion.choices[0].message.content
        
        if response_format!="json_object":
            response_json = {'response': str(response)}
        else:
            response_json = json_loads(response)
            
    except Exception as e:
        print(e)
        response_json = {
            'response': 'Sorry, I could not understand your request. Please try again.' if role == "receptionist" else "Sorry, there was an issue generating your response. Please try again."
        }

    return response_json
