'''
Converts the custom sentence transformer model to ONNX. 

Adapted from https://github.com/UKPLab/sentence-transformers/issues/46
'''

import onnxruntime
import torch
import transformers
from sentence_transformers import SentenceTransformer, models

class OnnxEncoder:
    '''
    OnxEncoder used to run SentenceTransformer model with domain adapter module under OnnxRuntime
    '''
    
    def __init__(self, session, tokenizer, pooling, adapter, normalization):
        self.session=session
        self.tokenizer=tokenizer
        self.max_length=tokenizer.__dict__["model_max_length"]
        self.pooling=pooling
        self.adapter=adapter
        self.normalization=normalization

    def encode(self, sentences: list):

        sentences = [sentences] if isinstance(sentences, str) else sentences
        
        # Tokenize the input sentences using the BERT tokenizer
        inputs = {
            k: v.numpy()
            for k, v in self.tokenizer(
                sentences,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).items()
        }
        
        # Run inference using the hidden transformer layer from the base model
        hidden_state = self.session.run(None, inputs)
        
        # Apply pooling layer
        sentence_embedding = self.pooling.forward(
            features={
                "token_embeddings": torch.Tensor(hidden_state[0]),
                "attention_mask": torch.Tensor(inputs.get("attention_mask")),
            },
        )
        
        # Apply the domain adapter layer
        sentence_embedding = self.adapter.forward(input_data=sentence_embedding)

        # Apply the normalization layer
        sentence_embedding = self.normalization.forward(features=sentence_embedding)

        sentence_embedding = sentence_embedding["sentence_embedding"]

        if sentence_embedding.shape[0] == 1:
            sentence_embedding = sentence_embedding[0]

        return sentence_embedding.numpy()


def sentence_transformers_onnx(
    model,
    output_path,
    config_path,
    do_lower_case=True,
    input_names=["input_ids", "attention_mask", "token_type_ids"],
    providers=["CUDAExecutionProvider"],
):
    '''
    Converts a sentence transformer model (SBERT) to a onnx model that can be run on OnnxRuntime
    
    Parameters
    ----------
    model
        SentenceTransformer model.
    path
        Model file dedicated to session inference.
    do_lower_case
        Either or not the model is cased.
    input_names
        Fields needed by the Transformer.
    providers
        Either run the model on CPU or GPU: ["CPUExecutionProvider", "CUDAExecutionProvider"].
    '''
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

    with torch.no_grad():

        symbolic_names = {0: "batch_size", 1: "max_seq_len"}

        torch.onnx.export(
            encoder,
            args=tuple(inputs.values()),
            f=f"{path}.onx",
            opset_version=14,
            do_constant_folding=True,
            input_names=input_names,
            output_names=["start", "end"],
            dynamic_axes={
                "input_ids": symbolic_names,
                "attention_mask": symbolic_names,
                "token_type_ids": symbolic_names,
                "start": symbolic_names,
                "end": symbolic_names,
            },
        )

        for modules in model.modules():
            for idx, module in enumerate(modules):
                if idx == 1:
                    pooling = module
                if idx == 2:
                    adapter = module
                if idx == 3:
                    normalization = module
            break

        return OnnxEncoder(
            session=onnxruntime.InferenceSession(f"{path}.onx", providers=providers),
            tokenizer=tokenizer,
            pooling=pooling,
            adapter=adapter,
            normalization=normalization,
        )