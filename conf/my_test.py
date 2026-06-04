conf = {
    "dataset": "musique",
    "test_samples": 5,

    "embed_name": "ollama:qwen3-embedding",
    #生成摘要的模型 
    "abs_name": "ollama:qwen2.5:7b",
    "qa_name": "ollama:qwen2.5:7b",

    "max_retrieval_time": 3,
    "tree_top_k": 5,
    # 启动混合检索方式--树状 dense retrieval + BM25 sparse retrieval + rerank / RRF 融合
    "hybrid_search": True,
    "sparse_top_k": 5,

    "save_dir": "./output/my_test",
}