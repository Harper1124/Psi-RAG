conf = {
    "read_local_pdf": "file",
    "embed_name": "qwen3-vl:Qwen/Qwen3-VL-Embedding-2B",
    "abs_name": "ollama:qwen3-vl:8b",
    "qa_name": "ollama:qwen3-vl:8b",
    "embed_model_kwargs": {
        "batch_size": 4,
        "normalize": True,
    },
    "force_split": True,
    "force_index_from_scratch": True,
    "force_sparse_index_from_scratch": True,
    "hybrid_search": False,
    "rerank": False,
    "tree_build_diagnostics": True,
    "data_dir": "/data/yzy/psiRAG/Psi-RAG/test.pdf",
    "read_local_pdf": "file",
}
