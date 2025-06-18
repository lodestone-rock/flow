# Flow Matching Trainer

A powerful training toolkit for image generation models using Flow Matching technique.

## Features

- Streaming dataloader that works directly from S3 or your local drive
- Flexible configuration via JSON
- Multi-GPU training support with automatic device detection
- Configurable inference during training
- Aim and Hugging Face integration
- Parameter efficient training with layer rotation and offloading

## Installation

```bash
# Clone the repository
git clone https://github.com/lodestone-rock/flow.git
cd flow

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

1. Prepare your training data in JSONL format (see Data Format section)
2. Configure your training settings in `training_config.json`
3. Run the training script:

```bash
python train_mp.py  # Automatically detects available GPUs
```

## Data Format

The trainer uses JSONL files for dataset metadata. Each line represents a single training example:

```json
{"filename": "https://your/s3/url", "caption_or_tags": "your image captions", "width": 816, "height": 1456, "is_tag_based": false, "is_url_based": true, "loss_weight": 0.5}
{"filename": "path/to/image/folder", "caption_or_tags": "tags, for, your, image", "width": 1024, "height": 1456, "is_tag_based": true, "is_url_based": false, "loss_weight": 0.3}
```

### JSONL Fields Explained:

- `filename`: Either an S3 URL or a local path to the image
- `caption_or_tags`: Text description or comma-separated tags for the image
- `width`/`height`: Image dimensions
- `is_tag_based`: Set to `true` if using tags instead of captions
- `is_url_based`: Set to `true` if the image is hosted on S3/URL
- `loss_weight`: Optional weighting factor for this sample (0.0-1.0)

## Configuration

The trainer is configured via a JSON file with the following sections:

### Training Settings

```json
"training": {
    "master_seed": 0,
    "cache_minibatch": 2,
    "train_minibatch": 1,
    "offload_param_count": 5000000000,
    "lr": 1e-05,
    "weight_decay": 0.0001,
    "warmup_steps": 1,
    "change_layer_every": 3,
    "trained_single_blocks": 2,
    "trained_double_blocks": 2,
    "save_every": 6,
    "save_folder": "checkpoints",
    "aim_path": "./training",
    "aim_experiment_name": "base",
    "aim_hash": null,
    "aim_steps": 0,
    "hf_repo_id": null,
    "hf_token": null
}
```

| Parameter | Description |
|-----------|-------------|
| `master_seed` | Random seed for reproducibility |
| `cache_minibatch` | T5 and VAE latent minibatch size (for text and image embedding generation) |
| `train_minibatch` | Model training minibatch size |
| `offload_param_count` | Frozen parameter count to offload to CPU during optimizer updates |
| `lr` | Learning rate |
| `weight_decay` | Weight decay for regularization |
| `warmup_steps` | Number of warmup steps for the learning rate scheduler |
| `change_layer_every` | Change randomly trained layers every X parameter updates |
| `trained_single_blocks` | Number of trainable transformer single blocks |
| `trained_double_blocks` | Number of trainable transformer double blocks |
| `save_every` | Save model checkpoint every X steps |
| `save_folder` | Directory to save model checkpoints |
| `aim_path` | Aim directory path (optional, Windows need to install by yourself) |
| `aim_experiment_name` | Aim experiment name (optional) |
| `aim_hash` | Aim hash (optional) |
| `aim_steps` | Aim steps (optional) |
| `hf_repo_id` | Hugging Face repository ID for pushing models (optional) |
| `hf_token` | Hugging Face API token (optional) |

### Inference Settings

```json
"inference": {
    "inference_every": 2,
    "inference_folder": "inference_folder",
    "steps": 20,
    "guidance": 3,
    "cfg": 1,
    "prompts": [
        "a cute cat sat on a mat while receiving a head pat from his owner called Matt",
        "baked potato, on the space floating orbiting around the earth"
    ],
    "first_n_steps_wo_cfg": -1,
    "image_dim": [512, 512],
    "t5_max_length": 512
}
```

| Parameter | Description |
|-----------|-------------|
| `inference_every` | Run inference every X training steps |
| `inference_folder` | Directory to save generated images |
| `steps` | Number of sampling steps for generation |
| `guidance` | Guidance scale for classifier-free guidance |
| `cfg` | Classifier-free guidance scale |
| `prompts` | List of text prompts for test generation |
| `first_n_steps_wo_cfg` | Number of initial steps without classifier-free guidance (-1 to disable) |
| `image_dim` | Output image dimensions [width, height] |
| `t5_max_length` | Maximum token length for text encoder |

You can add multiple inference configurations using the `extra_inference_config` array.

### Dataloader Settings

```json
"dataloader": {
    "batch_size": 8,
    "jsonl_metadata_path": "test_training_data.jsonl",
    "image_folder_path": "/images",
    "base_resolution": [256],
    "shuffle_tags": true,
    "tag_drop_percentage": 0.0,
    "uncond_percentage": 0.0,
    "resolution_step": 64,
    "num_workers": 2,
    "prefetch_factor": 2,
    "ratio_cutoff": 2.0,
    "thread_per_worker": 2
}
```

| Parameter | Description |
|-----------|-------------|
| `batch_size` | Total batch size per training step (must be divisible by number of GPUs) |
| `jsonl_metadata_path` | Path to the JSONL metadata file |
| `image_folder_path` | Base path for local images (if used) |
| `base_resolution` | Base training resolution for images (can be multi res) |
| `shuffle_tags` | Whether to randomly shuffle tags (for tag-based training) |
| `tag_drop_percentage` | Percentage of tags to randomly drop during training |
| `uncond_percentage` | Percentage of samples to use as unconditional samples |
| `resolution_step` | Step size for resolution buckets |
| `num_workers` | Number of dataloader workers |
| `prefetch_factor` | Prefetch factor for dataloader |
| `ratio_cutoff` | Maximum aspect ratio cutoff |
| `thread_per_worker` | Number of threads per worker |

### Model Settings

```json
"model": {
    "chroma_path": "models/flux/FLUX.1-schnell/chroma-8.9b.safetensors",
    "vae_path": "models/flux/ae.safetensors",
    "t5_path": "models/flux/text_encoder_2",
    "t5_config_path": "models/flux/text_encoder_2/config.json",
    "t5_tokenizer_path": "models/flux/tokenizer_2",
    "t5_to_8bit": true,
    "t5_max_length": 512
}
```

| Parameter | Description |
|-----------|-------------|
| `chroma_path` | Path to the base model checkpoint to resume from |
| `vae_path` | Path to the VAE model |
| `t5_path` | Path to the T5 text encoder model |
| `t5_config_path` | Path to the T5 config file |
| `t5_tokenizer_path` | Path to the T5 tokenizer |
| `t5_to_8bit` | Whether to load T5 in 8-bit precision to save memory |
| `t5_max_length` | Maximum token length for text encoder |

## Advanced Usage

### Parameter Efficient Training

The trainer supports efficient training by focusing on specific transformer blocks:

- `trained_single_blocks`: Number of single transformer blocks to train
- `trained_double_blocks`: Number of double transformer blocks to train
- `change_layer_every`: Frequency of rotating which layers are being trained

This approach allows training large models on limited hardware by only updating a subset of parameters at once.

### Multiple Inference Configurations

You can set up multiple inference configurations to test different settings during training:

```json
"extra_inference_config":[
    {
        "inference_every": 2,
        "inference_folder": "inference_folder_cfg4",
        "steps": 20,
        "guidance": 3,
        "cfg": 4,
        "prompts": [
            "a cute cat sat on a mat while receiving a head pat from his owner called Matt",
            "baked potato, on the space floating orbiting around the earth"
        ],
        "first_n_steps_wo_cfg": 0,
        "image_dim": [512, 512],
        "t5_max_length": 512
    }
]
```

## Example Complete Configuration

```json
{
    "training": {
        "master_seed": 0,
        "cache_minibatch": 2,
        "train_minibatch": 1,
        "offload_param_count": 5000000000,
        "lr": 1e-05,
        "weight_decay": 0.0001,
        "warmup_steps": 1,
        "change_layer_every": 3,
        "trained_single_blocks": 2,
        "trained_double_blocks": 2,
        "save_every": 6,
        "save_folder": "testing",
        "aim_path": "./training",
        "aim_experiment_name": "base",
        "aim_hash": null,
        "aim_steps": 0,
        "hf_repo_id": null,
        "hf_token": null
    },
    "inference": {
        "inference_every": 2,
        "inference_folder": "inference_folder",
        "steps": 20,
        "guidance": 3,
        "cfg": 1,
        "prompts": [
            "a cute cat sat on a mat while receiving a head pat from his owner called Matt",
            "baked potato, on the space floating orbiting around the earth"
        ],
        "first_n_steps_wo_cfg": -1,
        "image_dim": [512, 512],
        "t5_max_length": 512
    },
    "extra_inference_config":[
        {
            "inference_every": 2,
            "inference_folder": "inference_folder",
            "steps": 20,
            "guidance": 3,
            "cfg": 4,
            "prompts": [
                "a cute cat sat on a mat while receiving a head pat from his owner called Matt",
                "baked potato, on the space floating orbiting around the earth"
            ],
            "first_n_steps_wo_cfg": 0,
            "image_dim": [512, 512],
            "t5_max_length": 512
        }
    ],
    "dataloader": {
        "batch_size": 8,
        "jsonl_metadata_path": "test_training_data.jsonl",
        "image_folder_path": "/images",
        "base_resolution": [512],
        "shuffle_tags": true,
        "tag_drop_percentage": 0.0,
        "uncond_percentage": 0.0,
        "resolution_step": 64,
        "num_workers": 2,
        "prefetch_factor": 2,
        "ratio_cutoff": 2.0,
        "thread_per_worker": 100
    },
    "model": {
        "chroma_path": "models/flux/FLUX.1-schnell/chroma-8.9b.safetensors",
        "vae_path": "models/flux/ae.safetensors",
        "t5_path": "models/flux/text_encoder_2",
        "t5_config_path": "models/flux/text_encoder_2/config.json",
        "t5_tokenizer_path": "models/flux/tokenizer_2",
        "t5_to_8bit": true,
        "t5_max_length": 512
    }
}
```

## Troubleshooting

### Common Issues

1. **Out of Memory Errors**: Try reducing `batch_size`, increasing `offload_param_count`, or reducing `trained_single_blocks` and `trained_double_blocks`.

2. **Slow Training**: Check `num_workers` and `prefetch_factor` settings for your dataloader. Also consider increasing `cache_minibatch` if you have sufficient GPU memory.

3. **Poor Generation Quality**: Adjust `cfg` and `guidance` parameters in your inference settings, or increase the number of `steps`.

## License

**Apache 2.0**

## Citation
```
@misc{rock2025flow,
  author = {Lodestone Rock},
  title = {{Flow}},
  year = {2025},
  note = {Github repository},
  howpublished = {\url{https://github.com/lodestone-rock/flow}},
}
```