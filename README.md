# Domain Adapter for Sentence Transformer Models: Semantic Search

**Sentence Transformer Models** are transformer-based models that given an input sentence provide an _embedding_ that captures the semantics of the text such that any related sentences are clustered together in the space of the _embedding_. These models have significant potential to improve information retrieval and search.

Given a query, we can compute its textual _embeddings_ an then use measures of similarity such as cosine similarity to obtain the closest sentences in the corpus and provide the relevant search results.

This repository presents a sentence transformer model that was fine-tuned from a large [BERT-based model](https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2) with a search adaptor architecture that allows it to be slightly more performant on the domain of academic research publications than generic models.

## Dataset
To customize the model for the scientific domain, which encompasses publications, authors, and venues, we chose to utilize the REST API provided by Semantic Scholar. 
This API facilitates easy searches, with a limit of up to one request per second if an API key is available. The information offered by Semantic Scholar includes references and citations for each publication, as well as details about authors, journals, areas of study, PDF URLs, and more. 

We processed a sample of the resulting data to define a series of triples that link the retrieved entities (authors, publication venues, papers) through their respective properties, such as cited/referenced in, published in/by, written by, co-authored with, etc.

## Results
Our evaluation focused on evaluating the performance of domain-adapted base and custom models on entity retrieval tasks. We use several metrics, including hit rate, accuracy, precision, recall, mean reciprocal rank (MRR), normalized discounted cumulative gain (NDCG), and mean average precision (MAP), to comprehensively evaluate the effectiveness of the models.

The domain-adapted model consistently outperformed the base model on all metrics, demonstrating its superior performance in retrieving relevant entities from the dataset. Specifically, the hit rate for the custom model was 0.94, indicating that the relevant document was among the top 10 results in 94% of queries, compared to only 10% for the base model. 

Furthermore, qualitative analysis revealed that the custom domain-adapted model effectively captured additional semantic information related to each entity's area of ​​expertise. This improved semantic representation resulted in more accurate and contextually relevant retrieval results, contributing to the model's superior performance.

In conclusion, the custom domain-adapted model demonstrates significantly improved performance on entity retrieval tasks, making it a valuable tool for information retrieval in scientific networks. Its enhanced ability to capture domain-specific information and retain core capabilities underscores its potential to revolutionize information retrieval systems across multiple domains.

## Deployment

We used the [NVIDIA Triton Inference Server](https://github.com/triton-inference-server) to serve the model experimenting with Torch-Tensor RT quantization to improve both the model throughput and latency over the network.

The deployed model receives the input tensors from the tokenization and returns the corresponding embeddings. The Pytorch model was first exported to _ONNX_ format, and then leveraged NVidia server infrastructure to serve the model effortlessly.

The server can be started with the following command.

```sh
sudo docker run --gpus=all --rm -p8000:8000 -p8001:8001 -p8002:8002 -v/absolute_path_/model_repository:/models nvcr.io/nvidia/tritonserver:24.04-py3 tritonserver --model-repository=/models
```

For analyzing the throughput and latency of the system, we used Nvidia triton server SDK for perfomanze analysis. You can launch the docker image with this command

```sh
docker run -ti --rm --network=host --name triton-client nvcr.io/nvidia/tritonserver:24.04-py3-sdk
```

And then use this command to get the model performance when the input sequence has 50 tokens. These values can be adjusted as you see fit.

```sh
perf_analyzer -m domain_adapter --concurrency-range 1:4 -u http://host.docker.internal:8000 --shape attention_mask:1,50 --shape input_ids:1,50 --shape token_type_ids:1,50
```

## Project structure

The project contains the followin modules.

- `custom_adapter_module`: This module exports the AdaterModule, a _Pytorch_ class defining the architecture of the domain adapter.
- `onnx_converter`: This module converts the Pytorch-based sentence transformer model to Onnx format for interoperability.

### Adapted Model Deployment + RAG Agent + LLM + TelegramBot
This project demonstrates the deployment of an adapted sentence transformer model combined with a Retrieval-Augmented Generation (RAG) agent and a Large Language Model (LLM), integrated with a Telegram bot for efficient information retrieval and user interaction.

- `Adapted Model`: A fine-tuned BERT-based sentence transformer model optimized for the domain of academic research publications.
- `RAG Agent`: A Retrieval-Augmented Generation agent that combines the power of retrieval-based models with generative models to enhance the quality of the responses.
- `LLM`: A Large Language Model that provides advanced language understanding and generation capabilities.
- `Telegram Bot`: A bot that facilitates user interaction and retrieves information based on user queries through the Telegram messaging platform.

## How to run the notebooks?

The repository contains various Jupyter notebooks to perform various tasks.

- `model.ipynb` This notebook contains the model training steps and the analysis of its performance compared to the baseline model.
- `data_processing.ipynb` This notebook contains the steps for data preprocessing and preparation required for training the sentence transformer model and evaluating its effectiveness in the domain of academic research publications.
- `model_rag_agent.ipynb` This notebook contains the deployment steps for a Retrieval-Augmented Generation (RAG) agent, including model integration, performance evaluation, and deployment configuration for effective information retrieval in academic research publications.


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

## Additional resources

- `mlt_g4_paper.pdf` This document contains a detailed description of a domain-specific adapter for sentence transformer models, including the methodology, training process, and performance evaluation for enhancing semantic search capabilities in academic research publications.
- `mlt_g4_presentation.pdf` This presentation provides an overview of a domain-specific adapter for sentence transformer models, covering topics such as semantic search, domain adapters, dataset preparation, training process, results, and deployment, with a focus on enhancing semantic search capabilities in academic research publications.
-`[Video explaining the project](https://uniandes-my.sharepoint.com/:v:/r/personal/c_delarosap_uniandes_edu_co/Documents/MLTG4.mp4?csf=1&web=1&e=KCtrXe&nav=eyJyZWZlcnJhbEluZm8iOnsicmVmZXJyYWxBcHAiOiJTdHJlYW1XZWJBcHAiLCJyZWZlcnJhbFZpZXciOiJTaGFyZURpYWxvZy1MaW5rIiwicmVmZXJyYWxBcHBQbGF0Zm9ybSI6IldlYiIsInJlZmVycmFsTW9kZSI6InZpZXcifX0%3D)`