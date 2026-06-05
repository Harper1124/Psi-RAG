#!/usr/bin/env python
"""Inspect Psi-RAG tree pickle files.

Examples:
  python tools/inspect_tree_pkl.py output/test_pdf_..._tree.pkl
  python tools/inspect_tree_pkl.py output/test_pdf_..._tree.pkl --node-id 34
  python tools/inspect_tree_pkl.py output/test_pdf_..._tree.pkl --limit 10 --text-chars 500
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from collections import Counter
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class Node:
    """Small local stand-in for src.utils.Node, enough for inspection."""

    def __init__(
        self,
        text: str,
        index: int,
        document_index: int,
        chunk_index: int,
        children: set[int],
        embeddings: Any,
        metadata: dict[str, Any] | None = None,
        raw: Any = None,
    ) -> None:
        self.text = "" if text is None else str(text)
        self.index = index
        self.document_index = document_index
        self.chunk_index = chunk_index
        self.children = children
        self.embeddings = embeddings
        self.metadata = metadata or {}
        self.raw = raw


class Tree:
    """Small local stand-in for src.utils.Tree, enough for inspection."""

    def __init__(self, all_nodes: Any, root_nodes: Any, leaf_nodes: Any, layer_to_node_indices: Any) -> None:
        self.all_nodes = all_nodes
        self.root_nodes = root_nodes
        self.leaf_nodes = leaf_nodes
        self.layer_to_node_indices = layer_to_node_indices
        self.num_layers = max(layer_to_node_indices.keys()) if layer_to_node_indices else 0


class TreeUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> Any:
        if module == "src.utils" and name == "Node":
            return Node
        if module == "src.utils" and name == "Tree":
            return Tree
        return super().find_class(module, name)


def as_mapping(nodes: Any) -> dict[int, Any]:
    if isinstance(nodes, dict):
        return dict(sorted(nodes.items(), key=lambda item: int(item[0])))
    return {getattr(node, "index", i): node for i, node in enumerate(nodes or [])}


def load_tree(path: Path) -> Any:
    try:
        if path.is_dir():
            return load_tree_chunks(path)
        with path.open("rb") as file:
            return TreeUnpickler(file).load()
    except ModuleNotFoundError as exc:
        missing = exc.name or str(exc)
        raise SystemExit(
            f'Missing Python package "{missing}" while reading the pickle. '
            "Run this script inside the same Psi-RAG environment used to build the tree, "
            "or install the project requirements first."
        ) from exc


def load_tree_chunks(path: Path) -> Tree:
    manifest_path = path / "manifest.json"
    if not manifest_path.exists():
        raise ValueError(f"Bucketed tree directory is missing manifest.json: {path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    all_nodes = {}
    for chunk_file in manifest.get("chunk_files", []):
        chunk_path = path / chunk_file
        with chunk_path.open("rb") as file:
            node_chunk = TreeUnpickler(file).load()
        for idx, node_data in node_chunk.items():
            all_nodes[int(idx)] = deserialize_node(node_data)

    root_indices = [int(idx) for idx in manifest["root_node_indices"]]
    leaf_indices = [int(idx) for idx in manifest["leaf_node_indices"]]
    layer_to_node_indices = {
        int(layer): [int(idx) for idx in node_ids]
        for layer, node_ids in manifest["layer_to_node_indices"].items()
    }
    return Tree(
        all_nodes=all_nodes,
        root_nodes={idx: all_nodes[idx] for idx in root_indices},
        leaf_nodes={idx: all_nodes[idx] for idx in leaf_indices},
        layer_to_node_indices=layer_to_node_indices,
    )


def deserialize_node(data: dict[str, Any]) -> Node:
    return Node(
        text=data["text"],
        index=int(data["index"]),
        document_index=data["document_index"],
        chunk_index=data["chunk_index"],
        children=set(int(idx) for idx in data["children"]),
        embeddings=data["embeddings"],
        metadata=data.get("metadata", {}),
        raw=data.get("raw"),
    )


def short(value: Any, max_chars: int) -> str:
    text = "" if value is None else str(value)
    text = " ".join(text.split())
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def embedding_dim(node: Any) -> int | None:
    emb = getattr(node, "embeddings", None)
    if emb is None:
        return None
    shape = getattr(emb, "shape", None)
    if shape:
        return int(shape[-1])
    try:
        return len(emb)
    except TypeError:
        return None


def node_text(node: Any) -> str:
    raw = getattr(node, "raw", None)
    if isinstance(raw, dict):
        for key in ("text", "caption", "ocr", "table_body", "html"):
            if raw.get(key):
                return str(raw[key])
    return str(getattr(node, "text", ""))


def node_summary(node: Any, layer_by_node: dict[int, int] | None, text_chars: int) -> dict[str, Any]:
    node_id = int(getattr(node, "index", -1))
    children = sorted(getattr(node, "children", set()) or [])
    raw = getattr(node, "raw", None)
    metadata = getattr(node, "metadata", {}) or {}
    summary = {
        "id": node_id,
        "layer": None if layer_by_node is None else layer_by_node.get(node_id),
        "document_index": getattr(node, "document_index", None),
        "chunk_index": getattr(node, "chunk_index", None),
        "children_count": len(children),
        "children": children,
        "embedding_dim": embedding_dim(node),
        "text": short(node_text(node), text_chars),
    }
    if metadata:
        summary["metadata"] = metadata
    if isinstance(raw, dict):
        summary["raw_keys"] = sorted(raw.keys())
        summary["modality"] = raw.get("modality") or raw.get("type")
        summary["page"] = raw.get("page", raw.get("page_idx"))
    return summary


def get_tree_list(obj: Any) -> list[Any]:
    return obj if isinstance(obj, list) else [obj]


def layer_mapping(tree: Any) -> dict[int, int]:
    result = {}
    for layer, node_ids in getattr(tree, "layer_to_node_indices", {}).items():
        for node_id in node_ids:
            result[int(node_id)] = int(layer)
    return result


def print_stats(tree: Any, path: Path, tree_id: int | None) -> None:
    trees = get_tree_list(tree)
    selected = trees[tree_id] if tree_id is not None else trees[0]
    all_nodes = as_mapping(getattr(selected, "all_nodes", {}))
    root_nodes = as_mapping(getattr(selected, "root_nodes", {}))
    leaf_nodes = as_mapping(getattr(selected, "leaf_nodes", {}))
    layers = getattr(selected, "layer_to_node_indices", {}) or {}
    layer_counts = {int(layer): len(node_ids) for layer, node_ids in sorted(layers.items())}
    child_counts = [len(getattr(node, "children", set()) or []) for node in all_nodes.values()]
    modalities = Counter()
    for node in all_nodes.values():
        raw = getattr(node, "raw", None)
        if isinstance(raw, dict):
            modalities[str(raw.get("modality") or raw.get("type") or "unknown")] += 1

    print("=== Psi-RAG Tree ===")
    print(f"path: {path.resolve()}")
    print(f"size_mb: {path_size(path) / (1024 ** 2):.2f}")
    print(f"trees: {len(trees)}")
    if tree_id is not None:
        print(f"tree_id: {tree_id}")
    print(f"nodes: {len(all_nodes)}")
    print(f"roots: {list(root_nodes.keys())}")
    print(f"leaves: {len(leaf_nodes)}")
    print(f"layers: {len(layer_counts)}")
    print(f"nodes_per_layer: {layer_counts}")
    if child_counts:
        print(
            "children_per_node: "
            f"avg={sum(child_counts) / len(child_counts):.2f}, "
            f"min={min(child_counts)}, max={max(child_counts)}"
        )
    dims = sorted({dim for dim in (embedding_dim(node) for node in all_nodes.values()) if dim})
    print(f"embedding_dims: {dims}")
    if modalities:
        print(f"modalities: {dict(modalities)}")
    print()


def path_size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(file.stat().st_size for file in path.rglob("*") if file.is_file())


def dump_node(tree: Any, node_id: int, text_chars: int, show_raw: bool) -> None:
    nodes = as_mapping(getattr(tree, "all_nodes", {}))
    if node_id not in nodes:
        raise SystemExit(f"Node id {node_id} not found. Available range: {min(nodes)}..{max(nodes)}")

    node = nodes[node_id]
    layer_by_node = layer_mapping(tree)
    print(json.dumps(node_summary(node, layer_by_node, text_chars), ensure_ascii=False, indent=2))
    print("\n=== Full Text ===")
    print(node_text(node))
    if show_raw:
        print("\n=== Raw ===")
        print(json.dumps(make_json_safe(getattr(node, "raw", None)), ensure_ascii=False, indent=2))


def print_samples(tree: Any, limit: int, text_chars: int) -> None:
    nodes = as_mapping(getattr(tree, "all_nodes", {}))
    layer_by_node = layer_mapping(tree)
    by_layer: dict[int, list[int]] = {}
    for node_id, layer in layer_by_node.items():
        by_layer.setdefault(layer, []).append(node_id)

    print("=== Sample Nodes ===")
    if by_layer:
        for layer in sorted(by_layer):
            print(f"\n[layer {layer}]")
            for node_id in sorted(by_layer[layer])[:limit]:
                summary = node_summary(nodes[node_id], layer_by_node, text_chars)
                print(
                    f"id={summary['id']} doc={summary['document_index']} "
                    f"chunk={summary['chunk_index']} children={summary['children_count']} "
                    f"text={summary['text']}"
                )
    else:
        for node_id, node in list(nodes.items())[:limit]:
            summary = node_summary(node, None, text_chars)
            print(f"id={node_id} children={summary['children_count']} text={summary['text']}")


def make_json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): make_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [make_json_safe(v) for v in value]
    if isinstance(value, set):
        return sorted(make_json_safe(v) for v in value)
    if hasattr(value, "tolist"):
        value = value.tolist()
        if isinstance(value, list) and len(value) > 20:
            return value[:20] + [f"... truncated {len(value) - 20} items"]
        return make_json_safe(value)
    try:
        json.dumps(value)
        return value
    except TypeError:
        return repr(value)


def export_json(tree: Any, path: Path, tree_id: int | None, text_chars: int) -> None:
    trees = get_tree_list(tree)
    selected = trees[tree_id] if tree_id is not None else trees[0]
    layer_by_node = layer_mapping(selected)
    nodes = as_mapping(getattr(selected, "all_nodes", {}))
    payload = {
        "tree_id": tree_id,
        "nodes": [node_summary(node, layer_by_node, text_chars) for node in nodes.values()],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"JSON exported to {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect a Psi-RAG tree .pkl file or bucketed tree directory.")
    parser.add_argument("path", help="Path to *_tree.pkl, or a bucketed tree directory.")
    parser.add_argument("--tree-id", type=int, help="Tree index when the pickle contains a list of trees.")
    parser.add_argument("--node-id", type=int, help="Print one node in detail.")
    parser.add_argument("--limit", type=int, default=5, help="Sample nodes to show per layer.")
    parser.add_argument("--text-chars", type=int, default=220, help="Characters to show in node summaries.")
    parser.add_argument("--show-raw", action="store_true", help="With --node-id, also print raw multimodal payload.")
    parser.add_argument("--export-json", type=Path, help="Export node summaries to a JSON file.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = Path(args.path)
    if not path.exists():
        raise SystemExit(f"Path not found: {path}")

    tree = load_tree(path)
    trees = get_tree_list(tree)
    if args.tree_id is not None and not 0 <= args.tree_id < len(trees):
        raise SystemExit(f"--tree-id must be between 0 and {len(trees) - 1}")
    selected = trees[args.tree_id] if args.tree_id is not None else trees[0]

    print_stats(tree, path, args.tree_id)
    if args.node_id is not None:
        dump_node(selected, args.node_id, args.text_chars, args.show_raw)
    else:
        print_samples(selected, args.limit, args.text_chars)
    if args.export_json:
        export_json(tree, args.export_json, args.tree_id, args.text_chars)


if __name__ == "__main__":
    main()
