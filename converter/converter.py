'''
Converts the custom sentence transformer model to ONNX. 

Adapted from https://github.com/UKPLab/sentence-transformers/issues/46
'''

import onnxruntime
import torch
from torch import nn
import transformers
from sentence_transformers import SentenceTransformer, models

def sentence_transformers_onnx(
    model,
    output_path,
    config_path,
    do_lower_case=False,
    input_names=["input_ids", "attention_mask", "token_type_ids"],
    device="cpu",
):
    '''
    Converts a sentence transformer model (SBERT) to a onnx model that can be run on OnnxRuntime
    
    Parameters
    ----------
    model
        SentenceTransformer model.
    output_path
        Model file dedicated to session inference.
    config_path
        The input model must be saved at this path.
    do_lower_case
        Either or not the model is cased.
    input_names
        Fields needed by the Transformer.
    device
        Either run the model on CPU or GPU
    '''
    providers=["CUDAExecutionProvider"] if device.type == "cuda" else ["CPUExecutionProvider"]
    
    configuration = transformers.AutoConfig.from_pretrained(
        config_path, from_tf=False, local_files_only=True
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        config_path, do_lower_case=do_lower_case, from_tf=False, local_files_only=True
    )

    encoder = transformers.AutoModel.from_pretrained(
        config_path, from_tf=False, config=configuration, local_files_only=True
    )

    sample_sentences = ["article"]

    inputs = tokenizer(
        sample_sentences,
        padding=True,
        truncation=True,
        max_length=tokenizer.__dict__["model_max_length"],
        return_tensors="pt",
    )

    model.eval()

    for modules in model.modules():
        for idx, module in enumerate(modules):
            if idx == 1:
                pooling = module
            if idx == 2:
                adapter = module
            if idx == 3:
                normalization = module
        break

    class SentenceTransformerModel(nn.Module):
        def __init__(self):
            super(SentenceTransformerModel, self).__init__()
            # BERT base model
            self.encoder = encoder
            # Pooling layer
            self.pooling = pooling
            # Adapter module
            self.adapter = adapter
            # Normalization Layer
            self.normalization = normalization
        
        def forward(self, input_ids, attention_mask, token_type_ids):            
            hidden_state = self.encoder.forward(input_ids, attention_mask, token_type_ids)
            
            sentence_embedding = self.pooling.forward(
                features={
                    "token_embeddings": torch.Tensor(hidden_state[0]),
                    "attention_mask": torch.Tensor(attention_mask),
                },
            )

            sentence_embedding = self.adapter.forward(features=sentence_embedding)

            sentence_embedding = self.normalization.forward(features=sentence_embedding)

            sentence_embedding = sentence_embedding["sentence_embedding"]

            if sentence_embedding.shape[0] == 1:
                sentence_embedding = sentence_embedding[0]

            return sentence_embedding

    with torch.no_grad():
        inference_model = SentenceTransformerModel()
        inference_model.to(device)
        inference_model.eval()
        
        symbolic_names = {0: "batch_size", 1: "max_seq_len"}

        torch.onnx.export(
            inference_model,
            args=tuple(inputs.values()),
            f=f"{output_path}.onnx",
            opset_version=14,
            do_constant_folding=True,
            input_names=input_names,
            dynamic_axes={
                "input_ids": symbolic_names,
                "attention_mask": symbolic_names,
                "token_type_ids": symbolic_names,
            },
        )

        return inference_model

        return OnnxEncoder(
            session=onnxruntime.InferenceSession(f"{output_path}.onnx", providers=providers),
            tokenizer=tokenizer,
            pooling=pooling,
            adapter=adapter,
            normalization=normalization,
            device=device,
        )