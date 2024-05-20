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
<to do>

## How to run the notebooks?

The repository contains various Jupyter notebooks to perform various tasks.

- `model.ipynb` This notebook contains the model training steps and the analysis of its performance compared to the baseline model.
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
