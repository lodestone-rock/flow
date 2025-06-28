import gc
import os
import time
import json
import math
import logging
import warnings
import threading
from copy import deepcopy
from datetime import datetime

import torch
from safetensors.torch import save_file, load_file, safe_open
from huggingface_hub import HfApi, login

import logging

log = logging.getLogger(__name__)


def load_file_multipart(base_folder, device=None):
    state_dict = {}
    with open(os.path.join(base_folder, "model.safetensors.index.json")) as f:
        index = json.load(f)

    # List all .safetensors files in the folder
    for filename in os.listdir(base_folder):
        if filename.endswith(".safetensors"):
            file_path = os.path.join(base_folder, filename)

            # Load the safetensors file
            tensor_dict = load_file(file_path, device=device)

            # Append its contents to the combined dictionary
            state_dict.update(tensor_dict)

    if "param_count" in index["metadata"]:
        param_count = sum(t.numel() for t in state_dict.values())
        if index["metadata"]["param_count"] != param_count:
            warnings.warn(
                f"param count mismatched: loaded param count {param_count}, index metadata {index['param_count']}"
            )

    return state_dict


def save_file_multipart(
    state_dict, base_folder, metadata=None, num_shards=2, _json_index_only=False
):
    if not os.path.exists(base_folder):
        os.makedirs(base_folder)

    total_size = sum(t.numel() * t.element_size() for t in state_dict.values())
    target_size = math.ceil(total_size / num_shards)

    shards = [{} for _ in range(num_shards)]
    weight_map = {}

    size = 0
    shard_index = 0
    for key, tensor in state_dict.items():
        size += tensor.numel() * tensor.element_size()
        if target_size < size:
            shard_index += 1
            size = 0
        shards[shard_index][key] = tensor
        weight_map[key] = f"model-{shard_index+1:05d}-of-{num_shards:05d}.safetensors"

    if not _json_index_only:
        for i, shard in enumerate(shards):
            shard_filename = f"model-{i+1:05d}-of-{num_shards:05d}.safetensors"
            save_file(shard, os.path.join(base_folder, shard_filename))

    index = {"metadata": {"total_size": total_size}, "weight_map": weight_map}

    if metadata:
        index["metadata"].update(metadata)

    with open(os.path.join(base_folder, "model.safetensors.index.json"), "w") as f:
        json.dump(index, f, indent=2, ensure_ascii=False)

    return num_shards


def load_safetensors(file_path: str, device: str = "cpu"):
    statedict = {}

    with safe_open(file_path, framework="pt", device=device) as f:
        # Iterate over all keys and select layers that match the criteria
        for layer_name in f.keys():
            statedict[layer_name] = f.get_tensor(layer_name)

    return statedict


def load_selected_keys(filename, exclude_keywords=[]):
    """Loads all keys from a safetensors file except those containing specified keywords.

    Args:
        filename: Path to the safetensors file.
        exclude_keywords: List of keywords to exclude.

    Returns:
        A dictionary containing the loaded tensors.
    """

    tensors = {}
    with safe_open(filename, framework="pt") as f:
        for key in f.keys():
            if not any(keyword in key for keyword in exclude_keywords):
                tensors[key] = f.get_tensor(key)
    return tensors


def load_layers_by_keywords_from_safetensors(
    file_path: str,
    include_keywords: list[str],
    exclude_keywords: list[str] = None,
    device: str = "cpu",
):
    """
    Load specific layers from a .safetensors file based on keyword lists, including and excluding layers.

    Args:
        file_path (str): Path to the .safetensors file.
        include_keywords (list[str]): List of keywords to include in the search.
        exclude_keywords (list[str], optional): List of keywords to exclude from the search. Defaults to None.
        device (str): Device to load the tensors onto (e.g., "cpu" or "cuda").

    Returns:
        dict: A dictionary with layers that contain at least one include keyword and none of the exclude keywords in their names.
    """
    matched_layers_state_dict = {}

    with safe_open(file_path, framework="pt", device=device) as f:
        # Iterate over all keys and select layers that match the criteria
        for layer_name in f.keys():
            include_match = any(keyword in layer_name for keyword in include_keywords)
            exclude_match = any(
                keyword in layer_name for keyword in exclude_keywords or []
            )  # Handle empty exclude_keywords list
            if include_match and not exclude_match:
                matched_layers_state_dict[layer_name] = f.get_tensor(layer_name)

    if not matched_layers_state_dict:
        print(
            f"No layers found matching the include keywords {include_keywords} and not the exclude keywords {exclude_keywords} in the safetensors file."
        )
    else:
        print(
            f"Loaded {len(matched_layers_state_dict)} layers matching the include keywords {include_keywords} and not {exclude_keywords}."
        )

    return matched_layers_state_dict
