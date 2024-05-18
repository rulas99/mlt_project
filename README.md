# Domain Adapter for Sentence Transformer Models: Semantic Search

**Sentence Transformer Models** are transformer-based models that given an input sentence provide an _embedding_ that captures the semantics of the text such that any related sentences are clustered together in the space of the _embedding_. These models have significant potential to improve information retrieval and search.

Given a query, we can compute its textual _embeddings_ an then use measures of similarity such as cosine similarity to obtain the closest sentences in the corpus and provide the relevant search results.

This repository presents a sentence transformer model that was fine-tuned from a large BERT-based model with a search adaptor architecture that allows it to perform better on the domain of academic research publications.

## Dataset

## Results

## How to run the notebooks?

The repository contains various Jupyter notebooks to perform various tasks.

- `model.ipynb` This notebook contains the model architecture and training scripts.

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
