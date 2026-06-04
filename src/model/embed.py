import os
import logging

from abc import ABC, abstractmethod
from typing import Any, Dict, List
from tqdm import tqdm

import torch
import ollama
import numpy as np
import transformers
from torch.nn.functional import normalize
from openai import OpenAI
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
from tenacity import retry, stop_after_attempt, wait_random_exponential

logging.basicConfig(format="%(asctime)s - %(message)s", 
                    level=logging.INFO,
                    filename="./log/stdout.log",
                    filemode="a"
                    )


class BaseEmbeddingModel(ABC):
    model_name: str

    @abstractmethod
    def embed(self, text) -> np.ndarray:
        pass

    def __repr__(self):
        return self.model_name


class OpenAIEmbeddingModel(BaseEmbeddingModel):
    def __init__(self, model_name="text-embedding-ada-002", **kwargs):
        self.client = OpenAI()
        self.model_name = model_name

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def embed(self, text):
        text = text.replace("\n", " ")
        return (
            self.client.embeddings.create(input=[text], model=self.model_name)
            .data[0]
            .embedding
        )


class SentenceTransformersEmbeddingModel(BaseEmbeddingModel):
    def __init__(self, model_name="sentence-transformers/multi-qa-mpnet-base-cos-v1", cache_dir=None, **kwargs):
        """
        Args:
            model_name (str): sentence-transformers/multi-qa-mpnet-base-cos-v1, 
                              sentence-transformers/all-mpnet-base-v2,
                              nvidia/NV-Embed-v2,
                              Qwen/Qwen3-Embedding-0.6B,
                              Qwen/Qwen3-Embedding-8B,
        """
        self.model_name = model_name
        self.model = None
        self.cache_dir = cache_dir

    def load_model(self):
        if self.model is None:
            if self.model_name in ["nvidia/NV-Embed-v2"]:
                assert transformers.__version__ <= "4.46.0", "Use old transformers package (<= 4.46.0)."
                self.model = SentenceTransformer(self.model_name, 
                                                 trust_remote_code=True, 
                                                 cache_folder=self.cache_dir,
                                                 config_kwargs={'use_cache': False},
                                                 )
                self.model.max_seq_length = 32768
                self.model.tokenizer.padding_side="right"
            else:
                self.model = SentenceTransformer(self.model_name)

    def add_eos(self, input_examples):
        input_examples = [input_example + self.model.tokenizer.eos_token for input_example in input_examples]
        return input_examples

    def embed(self, text, **kwargs):
        self.load_model()
            
        if self.model_name in ("nvidia/NV-Embed-v2",):
            text = self.add_eos(text)

            kwargs.setdefault("batch_size", 8)
            kwargs.setdefault("prompt", "")
            kwargs.setdefault("normalize_embeddings", True)
        elif self.model_name in ("Qwen/Qwen3-Embedding-8B",):
            kwargs.setdefault("batch_size", 32)
            
        return self.model.encode(text, show_progress_bar=False, **kwargs)


class OllamaEmbeddingModel(BaseEmbeddingModel):
    def __init__(self, model_name="qwen3", cache_dir=None, **kwargs):
        """
        Args:
            model_name (str): qwen3-embedding:latest, 
        """
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.model_kwargs = kwargs
        self.dimensions = self.model_kwargs.pop("dimensions", None)
    
    def embed(self, text):
        params = {
            "input": text,
            "options": self.model_kwargs,
            "model": self.model_name,
            "keep_alive": '10m',
        }
        if self.dimensions is not None:
            params["dimensions"] = self.dimensions
        embs = ollama.embed(**params).embeddings[0]
        return embs


class VLLMEmbeddingModel(BaseEmbeddingModel):
    def __init__(self, model_name="Qwen/Qwen3-Embedding-8B", cache_dir=None, **kwargs):
        self.model_name = model_name
        self.model = None
        self.cache_dir = cache_dir
        self.model_kwargs = kwargs
        self.dimensions = self.model_kwargs.pop("dimensions", None)

    def load_model(self):
        if self.model is None:
            try:
                from vllm import LLM
            except ImportError as e:
                raise ImportError("vllm is not installed.") from e

            init_kwargs = self.model_kwargs.copy()
            if self.cache_dir is not None:
                init_kwargs["download_dir"] = self.cache_dir
            self.model = LLM(model=self.model_name, task="embed", **init_kwargs)

    def embed(self, text):
        self.load_model()

        embed_kwargs = {}
        if self.dimensions is not None:
            embed_kwargs["dimensions"] = self.dimensions

        embs = [output.outputs.embedding for output in self.model.embed(text, **embed_kwargs)]
        if isinstance(text, str):
            return embs[0]
        return np.asarray(embs)


class Qwen3VLEmbeddingModel(BaseEmbeddingModel):
    def __init__(self, model_name="Qwen/Qwen3-VL-Embedding", cache_dir=None, **kwargs):
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.model = None
        self.processor = None
        self.embedder = None
        self.sentence_model = None
        self.model_kwargs = kwargs
        self.batch_size = self.model_kwargs.pop("batch_size", 4)
        self.normalize = self.model_kwargs.pop("normalize", True)

    def load_model(self):
        if self.embedder is not None or self.sentence_model is not None or self.model is not None:
            return

        try:
            from qwen3_vl_embedding import Qwen3VLEmbedder
            self.embedder = Qwen3VLEmbedder(
                model_name=self.model_name,
                cache_dir=self.cache_dir,
                **self.model_kwargs,
            )
            return
        except ImportError:
            pass

        try:
            self.sentence_model = SentenceTransformer(
                self.model_name,
                cache_folder=self.cache_dir,
                trust_remote_code=True,
            )
            return
        except Exception as sentence_transformer_error:
            self.sentence_transformer_error = sentence_transformer_error

        try:
            from transformers import AutoProcessor, AutoModel
        except ImportError as e:
            raise ImportError(
                "Qwen3-VL-Embedding requires either the official qwen3_vl_embedding package "
                "or a transformers version that can load the model."
            ) from e

        model_kwargs = self.model_kwargs.copy()
        model_kwargs.setdefault("trust_remote_code", True)
        model_kwargs.setdefault("device_map", "auto")
        model_kwargs.setdefault("torch_dtype", "auto")
        self.processor = AutoProcessor.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir,
            trust_remote_code=True,
        )
        self.model = AutoModel.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir,
            **model_kwargs,
        )

    def _normalize_input(self, item: str | Dict[str, Any]) -> Dict[str, Any]:
        if isinstance(item, dict):
            text = item.get("text") or item.get("caption") or item.get("ocr") or ""
            image = item.get("image") or item.get("image_path") or item.get("path")
            return {"text": text, "image": image}
        return {"text": "" if item is None else str(item), "image": None}

    def _embed_with_official_embedder(self, items: List[Dict[str, Any]]) -> np.ndarray:
        if hasattr(self.embedder, "encode"):
            return np.asarray(self.embedder.encode(items, batch_size=self.batch_size, normalize=self.normalize))
        if hasattr(self.embedder, "embed"):
            return np.asarray(self.embedder.embed(items, batch_size=self.batch_size, normalize=self.normalize))
        raise AttributeError("Qwen3VLEmbedder must expose encode() or embed().")

    def _embed_with_transformers(self, items: List[Dict[str, Any]]) -> np.ndarray:
        try:
            from PIL import Image
        except ImportError as e:
            raise ImportError("Pillow is required for Qwen3-VL image embedding.") from e

        embs = []
        for item in items:
            processor_kwargs = {
                "text": [item["text"]],
                "padding": True,
                "return_tensors": "pt",
            }
            image_path = item.get("image")
            if image_path:
                processor_kwargs["images"] = [Image.open(image_path).convert("RGB")]
            inputs = self.processor(**processor_kwargs)
            if hasattr(self.model, "device"):
                inputs = {k: v.to(self.model.device) for k, v in inputs.items() if hasattr(v, "to")}
            with torch.no_grad():
                outputs = self.model(**inputs)
                if hasattr(outputs, "embeddings"):
                    batch_embs = outputs.embeddings
                elif hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                    batch_embs = outputs.pooler_output
                else:
                    batch_embs = outputs.last_hidden_state[:, 0]
                if self.normalize:
                    batch_embs = normalize(batch_embs, p=2, dim=1)
                embs.append(batch_embs.float().detach().cpu().numpy())
        return np.concatenate(embs, axis=0)

    def embed(self, text):
        self.load_model()
        is_single = not isinstance(text, list)
        items = [self._normalize_input(text)] if is_single else [self._normalize_input(item) for item in text]

        if self.embedder is not None:
            embs = self._embed_with_official_embedder(items)
        elif self.sentence_model is not None:
            sentence_inputs = []
            for item in items:
                if item.get("image") and item.get("text"):
                    sentence_inputs.append({"text": item["text"], "image": item["image"]})
                elif item.get("image"):
                    sentence_inputs.append(item["image"])
                else:
                    sentence_inputs.append(item["text"])
            embs = self.sentence_model.encode(
                sentence_inputs,
                batch_size=self.batch_size,
                normalize_embeddings=self.normalize,
                show_progress_bar=False,
            )
        else:
            embs = self._embed_with_transformers(items)

        return embs[0] if is_single else embs


class TransformersEmbeddingModel(BaseEmbeddingModel):
    def __init__(self, model_name="facebook/contriever", cache_dir=None, **kwargs):
        """
        Args:
            model_name (str): facebook/contriever, 
                              nvidia/NV-Embed-v2 (one GPU only), 
        """
        self.model_name = model_name
        self.model = None
        self.model_kwargs = kwargs
        self.tokenizer = None
        self.cache_dir = cache_dir
        if self.model_name in ("nvidia/NV-Embed-v2"):
            assert transformers.__version__ <= "4.46.0", "Use old transformers package (<= 4.46.0)."

    def load_model(self):
        if self.model is None:
            model_init_params = {
                "trust_remote_code": True,
                'device_map': "auto",  # added this line to use multiple GPUs
                "torch_dtype": "auto",
            }
            model_kwargs = self.model_kwargs.copy()
            model_kwargs.update(model_init_params)
            self.model = AutoModel.from_pretrained(self.model_name, 
                                                   mirror=os.environ["HF_ENDPOINT"], 
                                                   cache_dir=self.cache_dir,
                                                   **model_kwargs)
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, 
                                                           mirror=os.environ["HF_ENDPOINT"],
                                                           cache_dir=self.cache_dir)

    def embed(self, text, **kwargs):
        self.load_model()
        if self.model_name in ("facebook/contriever"):
            return self._embed_contriever(text, **kwargs)
        elif self.model_name in ("nvidia/NV-Embed-v2"):
            return self._embed_nvidia(text, **kwargs)

    def _embed_contriever(self, text, **kwargs):
        kwargs.setdefault("normalize", True)
        kwargs.setdefault("output_hidden_states", False)

        inputs = self.tokenizer(text, padding=True, truncation=True, return_tensors='pt')
        outputs = self.model(**inputs)

        return outputs
    
    def _embed_nvidia(self, text: List[str], **kwargs):
        
        if isinstance(text, str):
            text = [text]
        kwargs.setdefault("instruction", "")
        kwargs.setdefault("batch_size", 4)
        kwargs.setdefault("num_workers", 32)
        kwargs.setdefault("norm", True)

        if len(text) <= kwargs["batch_size"]:
            kwargs["prompts"] = text 
            embs = self.model.encode(**kwargs)
        else:
            embs = []
            bar = tqdm(range(0, len(text), kwargs["batch_size"]), desc="creating leaf nodes")
            for i in bar:
                kwargs["prompts"] = text[i:i + kwargs["batch_size"]]
                embs.append(self.model.encode(**kwargs))
            bar.close()
            embs = torch.cat(embs, dim=0)
        
        if kwargs["norm"]:
            embs = normalize(embs, p=2, dim=1)

        return embs.squeeze().cpu().numpy()

