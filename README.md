# Domain Adapter for Sentence Transformer Models: Semantic Search

**Sentence Transformer Models** are transformer-based models that given an input sentence provide an _embedding_ that captures the semantics of the text such that any related sentences are clustered together in the space of the _embedding_. These models have significant potential to improve information retrieval and search.

Given a query, we can compute its textual _embeddings_ an then use measures of similarity such as cosine similarity to obtain the closest sentences in the corpus and provide the relevant search results.

This repository presents a sentence transformer model that was fine-tuned from a large [BERT-based model](https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2) with a search adaptor architecture that allows it to be slightly more performant on the domain of academic research publications than generic models.

## Dataset
<to do (agregar informacion del trabajo escrito)>
- data directory contains...

## Results
<to do (agregar informacion del trabajo escrito)>
- results directory contains....

## Deployment

We used the [NVIDIA Triton Inference Server](https://github.com/triton-inference-server) to serve the model experimenting with both Torch-Tensor RT and quantization to improve both the model throughput and latency over the network.

### Adapted Model Deployment + RAG Agent + LLM + TelegramBot
<to do>

## How to run the notebooks?

The repository contains various Jupyter notebooks to perform various tasks.

- `model.ipynb` This notebook contains the model architecture and training scripts.
- `data_processing.ipynb` <to do>
- `deploy.ipynb` <to do>
- `model_rag_agent.ipynb` <to do>

Please follow these steps to run any of the notebooks.

- Check if both Python 3.9 and Jupyter lab are correctly installed.

- We recommend creating a virtual environment named `env` for the project.

On Linux/MacOS, run:

```sh
python3 -m venv env
```

On Windows, run:

```sh
python -m venv env
```

- Activate the virtual environment.

On Linux/MacOS, run:

```sh
source env/bin/activate
```

On Windows, run:

```sh
.\env\Scripts\activate
```

- Install required packages

```sh
pip install -r requirements.txt
```

- Launch Jupyter Lab

```sh
jupyter-lab <notebook-name>
```
