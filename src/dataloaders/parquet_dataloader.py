"""Parquet-based dataloader for multi-source image-text datasets.

Supports:
- Loading from Hive-partitioned parquet directories (source=xxx/part-*.parquet)
- Per-source subsampling with optional repeat + warning
- Multi-caption sampling with configurable probabilities
- Tag-based vs caption-based flag per caption column
- Bucketing, round-robin multi-GPU sharding, reference images
- Same return signature as TextImageDataset

Example config snippet:
{
    "parquet_sources": {
        "e621":       {"path": "/data/parquet/source=e621",       "n_samples": 1000000},
        "deviantart": {"path": "/data/parquet/source=deviantart", "n_samples": 500000}
    },
    "caption_columns": {
        "tags":                    {"weight": 0.2, "is_tag_based": true},
        "midjourney_style_summary": {"weight": 0.3, "is_tag_based": false},
        "brief_summary":           {"weight": 0.5, "is_tag_based": false}
    },
    "filename_column":    "url",
    "width_column":       "image_width",
    "height_column":      "image_height",
    "loss_weight_column": null,          // or e.g. "sampling_probability"
    "image_folder_path":  "",            // base dir for local files; ignored for URLs
    ...
}
"""

import os
import math
import random
import logging
import warnings
from io import BytesIO

import concurrent.futures as _cf

import torch
from torch.utils.data import Dataset
import torchvision.transforms.v2 as v2
from PIL import Image
import pyarrow as pa
import pyarrow.parquet as pq
import psutil

from .bucketing_logic import (
    _bucket_generator,
    _normalize_width_height,
    _closest_bucket,
)
from . import color_profile_handling

try:
    import requests
    _REQUESTS_AVAILABLE = True
except ImportError:
    _REQUESTS_AVAILABLE = False

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _collect_parquet_files(path: str) -> list[str]:
    """Walk *path* and return all .parquet file paths, sorted."""
    if os.path.isfile(path) and path.endswith(".parquet"):
        return [path]
    files = []
    for root, _dirs, fnames in os.walk(path):
        for f in sorted(fnames):
            if f.endswith(".parquet"):
                files.append(os.path.join(root, f))
    if not files:
        raise FileNotFoundError(f"No parquet files found under: {path}")
    return files


def _read_parquet_columns(filepath: str, columns: list[str]) -> pa.Table:
    """Read only *columns* from a single parquet file."""
    # Use the file-level schema to skip missing columns gracefully
    schema = pq.read_schema(filepath)
    existing = [c for c in columns if c in schema.names]
    return pq.read_table(filepath, columns=existing)


def _cast_strings_to_large(table: pa.Table) -> pa.Table:
    """Cast all string/binary columns to large_string/large_binary.

    PyArrow's string type uses 32-bit offsets (max ~2 GB per column chunk).
    When concatenating many files the offset can overflow; large_string uses
    64-bit offsets and avoids the error entirely.
    """
    new_fields = []
    new_cols = []
    for i, field in enumerate(table.schema):
        col = table.column(i)
        if pa.types.is_string(field.type) or pa.types.is_large_string(field.type):
            col = col.cast(pa.large_utf8())
            field = field.with_type(pa.large_utf8())
        elif pa.types.is_binary(field.type) or pa.types.is_large_binary(field.type):
            col = col.cast(pa.large_binary())
            field = field.with_type(pa.large_binary())
        new_fields.append(field)
        new_cols.append(col)
    return pa.table({f.name: c for f, c in zip(new_fields, new_cols)})


def _load_parquet_source(
    path: str,
    n_samples: int | None,
    seed: int,
    columns: list[str],
    num_threads: int = 8,
) -> pa.Table:
    """Load parquet files under *path* in parallel, project *columns*, subsample.

    Returns a PyArrow Table — no pandas involved.
    Uses its own seeded RNG so callers can run this in parallel safely.
    """
    rng = random.Random(seed)
    parquet_files = _collect_parquet_files(path)

    # Parallel reads — each worker reads one file
    with _cf.ThreadPoolExecutor(max_workers=min(num_threads, len(parquet_files))) as ex:
        futures = {ex.submit(_read_parquet_columns, f, columns): f for f in parquet_files}
        tables = []
        for fut in _cf.as_completed(futures):
            tables.append(fut.result())

    # Cast strings to large_string before concat to avoid 32-bit offset overflow
    tables = [_cast_strings_to_large(t) for t in tables]

    # Concatenate with zero-copy where possible
    table = pa.concat_tables(tables, promote_options="default")
    del tables
    total = len(table)

    if n_samples is not None:
        if n_samples > total:
            warnings.warn(
                f"Requested {n_samples} samples from '{path}' but only {total} rows "
                f"are available. Repeating rows to reach the target.",
                stacklevel=3,
            )
            repeats = math.ceil(n_samples / total)
            idx = list(range(total)) * repeats
            rng.shuffle(idx)
            idx = idx[:n_samples]
        else:
            idx = rng.sample(range(total), n_samples)
        table = table.take(pa.array(idx, type=pa.int64()))

    return table


def _build_standardized_buckets(base_resolution: list[int], ratio_cutoff: float, step: int) -> dict:
    """Return {res: {(norm_w, norm_h): (w, h), ...}, ...}."""
    standardized_buckets = {}
    for res in base_resolution:
        buckets = _bucket_generator(res, ratio_cutoff, step)
        this_std = {}
        for bucket in buckets:
            b_st = _normalize_width_height(*bucket)
            this_std[b_st] = bucket
        standardized_buckets[res] = this_std
    return standardized_buckets


def _assign_bucket(image_width: int, image_height: int,
                   standardized_buckets: dict, ratio_cutoff: float,
                   rng: random.Random) -> tuple | None:
    """Return a randomly chosen bucket (w, h) for the given image dimensions, or None if filtered."""
    if image_width <= 0 or image_height <= 0:
        return None
    aspect_ratio = image_width / image_height
    if not (1.0 / ratio_cutoff < aspect_ratio < ratio_cutoff):
        return None
    w, h = _normalize_width_height(image_width, image_height)
    chosen_std = rng.choice(list(standardized_buckets.values()))
    return _closest_bucket(w, h, chosen_std)


# ---------------------------------------------------------------------------
# Main Dataset
# ---------------------------------------------------------------------------

class ParquetTextImageDataset(Dataset):
    """Multi-source parquet dataset with bucketing and multi-GPU sharding.

    Parameters
    ----------
    batch_size : int
        Global batch size (across all GPUs).
    parquet_sources : dict
        ``{source_name: {"path": str, "n_samples": int | None}}``
        ``n_samples`` is optional; omit or set to ``null`` to use all rows.
    caption_columns : dict
        ``{col_name: {"weight": float, "is_tag_based": bool}}``
        Weights are normalised internally.
    filename_column : str
        Column that holds the image filename or URL.
    width_column : str
        Column for image width.
    height_column : str
        Column for image height.
    loss_weight_column : str | None
        Column for per-sample loss weight. If ``None``, defaults to 1.
    image_folder_path : str
        Base directory prepended to local filenames. Ignored for URLs.
    base_res : list[int]
        Base resolutions for bucket generation.
    ratio_cutoff : float
        Maximum aspect ratio (images outside ``[1/r, r]`` are dropped).
    resolution_step : int
        Step size for bucket generation.
    shuffle_tags : bool
        Shuffle comma-separated tags when ``is_tag_based`` is True.
    tag_drop_percentage : float
        Fraction of tags that may be randomly dropped (0 = keep all).
    uncond_percentage : float
        Probability of replacing caption with empty string (CFG dropout).
    seed : int
        Master random seed.
    rank : int
        This GPU's rank (0-indexed).
    num_gpus : int
        Total number of GPUs.
    timeout : int
        HTTP request timeout in seconds.
    thread_per_worker : int
        Thread pool size for concurrent image loading.
    dummy_image : bool
        If True, skip actual image loading (for debugging).
    offset : int
        Skip the first *offset* batches.
    num_reference_images : int | None
        If set, load this many reference images per sample from the
        ``reference_images`` column (list of filenames/URLs).
    """

    def __init__(
        self,
        batch_size: int,
        parquet_sources: dict,
        caption_columns: dict,
        filename_column: str = "url",
        width_column: str = "image_width",
        height_column: str = "image_height",
        loss_weight_column: str | None = None,
        image_folder_path: str = "",
        base_res: list[int] = None,
        ratio_cutoff: float = 2.0,
        resolution_step: int = 64,
        shuffle_tags: bool = True,
        tag_drop_percentage: float = 0.1,
        uncond_percentage: float = 0.05,
        seed: int = 0,
        rank: int = 0,
        num_gpus: int = 1,
        timeout: int = 10,
        thread_per_worker: int = 100,
        dummy_image: bool = False,
        offset: int = 0,
        num_reference_images: int | None = None,
    ):
        if base_res is None:
            base_res = [1024]

        assert batch_size % num_gpus == 0, "batch_size must be divisible by num_gpus"

        self.batch_size = batch_size
        self.rank_batch_size = batch_size // num_gpus
        self.parquet_sources = parquet_sources
        self.filename_column = filename_column
        self.width_column = width_column
        self.height_column = height_column
        self.loss_weight_column = loss_weight_column
        self.image_folder_path = image_folder_path
        self.base_res = base_res
        self.ratio_cutoff = ratio_cutoff
        self.resolution_step = resolution_step
        self.shuffle_tags = shuffle_tags
        self.tag_drop_percentage = tag_drop_percentage
        self.uncond_percentage = uncond_percentage
        self.rank = rank
        self.num_gpus = num_gpus
        self.timeout = timeout
        self.thread_per_worker = thread_per_worker
        self.dummy_image = dummy_image
        self.num_reference_images = num_reference_images
        self.offset = offset

        # Normalise caption column weights
        self.caption_columns = caption_columns
        total_w = sum(v["weight"] for v in caption_columns.values())
        self._caption_col_names: list[str] = []
        self._caption_col_weights: list[float] = []
        self._caption_col_is_tag: list[bool] = []
        for col, cfg in caption_columns.items():
            self._caption_col_names.append(col)
            self._caption_col_weights.append(cfg["weight"] / total_w)
            self._caption_col_is_tag.append(bool(cfg.get("is_tag_based", False)))

        self._rng = random.Random(seed)

        self.image_transforms = v2.Compose(
            [v2.ToTensor(), v2.Normalize(mean=[0.5], std=[0.5])]
        )

        if _REQUESTS_AVAILABLE:
            import requests as _req
            self.session = _req.Session()
        else:
            self.session = None

        self.batches = self._load_batches()
        self._round_robin()
        self.batches = self.batches[offset:]

    # ------------------------------------------------------------------
    # Batch preparation
    # ------------------------------------------------------------------

    def _load_batches(self) -> list:
        rng = self._rng

        # Columns we actually need — avoids pulling heavy unused columns into RAM
        needed_cols = list({
            self.filename_column,
            self.width_column,
            self.height_column,
            *self._caption_col_names,
            *([] if not self.loss_weight_column else [self.loss_weight_column]),
            *([] if self.num_reference_images is None else ["reference_images"]),
        })

        # ---- 1. Load & subsample each source in parallel ----
        n_io_threads = min(8, psutil.cpu_count(logical=False) or 4)

        def _load_source(item):
            source_name, src_cfg = item
            path = src_cfg["path"]
            n_samples = src_cfg.get("n_samples", None)
            # Derive a deterministic but unique seed per source so parallel
            # calls don't share RNG state
            source_seed = rng.randint(0, 2**31)
            table = _load_parquet_source(path, n_samples, source_seed, needed_cols, n_io_threads)
            print(f"  [{source_name}] loaded {len(table):,} rows")
            return table

        with _cf.ThreadPoolExecutor(max_workers=len(self.parquet_sources)) as ex:
            tables = list(ex.map(_load_source, self.parquet_sources.items()))

        combined = pa.concat_tables(tables, promote_options="default")
        del tables
        total_rows = len(combined)
        print(f"Total rows after subsampling: {total_rows:,}")

        # ---- 2. Build bucket lookup ----
        standardized_buckets = _build_standardized_buckets(
            self.base_res, self.ratio_cutoff, self.resolution_step
        )
        std_bucket_list = list(standardized_buckets.values())  # for rng.choice

        # ---- 3. Extract columns as Python lists once — avoids per-row PyArrow overhead ----
        col_names = combined.schema.names

        def _col(name):
            if name in col_names:
                return combined.column(name).to_pylist()
            return None

        filenames    = _col(self.filename_column)
        widths       = _col(self.width_column)
        heights      = _col(self.height_column)
        loss_weights = _col(self.loss_weight_column) if self.loss_weight_column else None
        ref_images   = _col("reference_images") if self.num_reference_images is not None else None
        caption_cols = {c: _col(c) for c in self._caption_col_names}

        # Free the Arrow table — all data is now in plain Python lists
        del combined

        # ---- 4. Assign buckets and build sample dicts ----
        buckets: dict[tuple, list] = {}
        skipped = 0
        skipped_no_caption = 0

        for i in range(total_rows):
            try:
                w = int(widths[i])
                h = int(heights[i])
            except (TypeError, ValueError):
                skipped += 1
                continue

            bucket = _assign_bucket(w, h, standardized_buckets, self.ratio_cutoff, rng)
            if bucket is None:
                skipped += 1
                continue

            # Find which caption columns have a non-empty value for this row
            available_indices = [
                idx for idx, col_name in enumerate(self._caption_col_names)
                if caption_cols[col_name] is not None
                and caption_cols[col_name][i] is not None
                and str(caption_cols[col_name][i]).strip() != ""
            ]

            # Drop the row entirely if no caption column has a value
            if not available_indices:
                skipped_no_caption += 1
                continue

            # Sample only from available columns, re-normalising weights on the fly
            available_weights = [self._caption_col_weights[idx] for idx in available_indices]
            col_idx   = rng.choices(available_indices, weights=available_weights, k=1)[0]
            col_name  = self._caption_col_names[col_idx]
            is_tag    = self._caption_col_is_tag[col_idx]
            caption   = str(caption_cols[col_name][i]).strip()

            lw = 1.0
            if loss_weights is not None:
                try:
                    lw = float(loss_weights[i])
                except (TypeError, ValueError):
                    lw = 1.0

            filename = str(filenames[i])
            sample = {
                "filename":       filename,
                "caption_or_tags": caption,
                "bucket":         bucket,
                "is_tag_based":   is_tag,
                "is_url_based":   self._is_url(filename),
                "loss_weight":    lw,
                "reference_images": (ref_images[i] if ref_images and isinstance(ref_images[i], list) else []),
            }

            if bucket in buckets:
                buckets[bucket].append(sample)
            else:
                buckets[bucket] = [sample]

        if skipped:
            log.warning(f"Skipped {skipped:,} rows (bad dimensions or extreme aspect ratio)")
        if skipped_no_caption:
            log.warning(f"Skipped {skipped_no_caption:,} rows (all caption columns were None/empty)")

        # ---- 5. Shuffle within each bucket ----
        for key in buckets:
            rng.shuffle(buckets[key])

        post_count = sum(len(v) for v in buckets.values())
        print(f"There are {post_count:,} text-image pairs across {len(buckets)} buckets.")

        # ---- 6. Pack into full batches ----
        batches = []
        for b in buckets.values():
            samples = []
            for s in b:
                samples.append(s)
                if len(samples) == self.batch_size:
                    batches.append(samples)
                    samples = []
            # drop incomplete tail batch

        print(f"We got {len(batches):,} batches.")
        rng.shuffle(batches)
        return batches

    def _round_robin(self):
        """Slice batches for this rank (each batch → rank's slice)."""
        subset = []
        for batch in self.batches:
            subset.append(
                batch[
                    self.rank * self.rank_batch_size:
                    self.rank * self.rank_batch_size + self.rank_batch_size
                ]
            )
        self.batches = subset

    def resample(self):
        """Draw a fresh subsample from the parquet sources and rebuild batches.

        Call this at the end of each epoch so the per-source subsampling picks
        a different random subset next time.  The internal RNG advances with
        every call, so successive epochs are always distinct.

        Example::

            for epoch in range(num_epochs):
                for batch in dataloader:
                    ...
                dataset.resample()   # refresh before next epoch
        """
        print(f"[ParquetTextImageDataset] resampling (rank {self.rank})...")
        self.batches = self._load_batches()
        self._round_robin()
        self.batches = self.batches[self.offset:]
        print(f"[ParquetTextImageDataset] resample done — {len(self.batches)} batches available")

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    def _is_url(path: str) -> bool:
        return path.startswith("http://") or path.startswith("https://")

    @staticmethod
    def scale_and_crop_long_axis(image: Image.Image, target_height: int, target_width: int) -> Image.Image:
        if (target_width / target_height) >= (image.width / image.height):
            image = v2.functional.resize(
                image,
                [round(target_width * image.height / image.width), target_width],
                interpolation=v2.InterpolationMode.LANCZOS,
            )
        else:
            image = v2.functional.resize(
                image,
                [target_height, round(image.width * target_height / image.height)],
                interpolation=v2.InterpolationMode.LANCZOS,
            )
        return v2.functional.center_crop(image, [target_height, target_width])

    @staticmethod
    def _sample_elements_by_percentage(my_list: list, percentage: float) -> list:
        if not 0 <= percentage <= 1:
            raise ValueError("Percentage must be between 0 and 1")
        sample_size = math.ceil(len(my_list) * percentage)
        return random.sample(my_list, sample_size)

    @staticmethod
    def dummy_collate_fn(batch):
        return batch

    def __len__(self) -> int:
        return len(self.batches)

    # ------------------------------------------------------------------
    # Image loading
    # ------------------------------------------------------------------

    def _load_image(self, sample: dict) -> Image.Image | None:
        try:
            if self._is_url(sample["filename"]):
                if self.session is None:
                    raise RuntimeError("requests is not installed; cannot load URLs")
                response = self.session.get(sample["filename"], timeout=self.timeout)
                response.raise_for_status()
                return Image.open(BytesIO(response.content)).convert("RGB")
            else:
                image_path = os.path.join(self.image_folder_path, sample["filename"])
                jxl_path = os.path.splitext(image_path)[0] + ".jxl"
                if os.path.exists(jxl_path):
                    return color_profile_handling.open_srgb(jxl_path).convert("RGB")
                elif os.path.exists(image_path):
                    return Image.open(image_path).convert("RGB")
                else:
                    base, _ = os.path.splitext(sample["filename"])
                    for ext in ("png", "jpg", "jpeg", "webp"):
                        alt = os.path.join(self.image_folder_path, f"{base}.{ext}")
                        if os.path.exists(alt):
                            return Image.open(alt).convert("RGB")
            return None
        except Exception as e:
            log.error(f"Error loading image '{sample['filename']}' on rank {self.rank}: {e}")
            return None

    def _load_reference_image(self, ref_filename: str) -> Image.Image | None:
        try:
            if self._is_url(ref_filename):
                if self.session is None:
                    raise RuntimeError("requests is not installed; cannot load URLs")
                response = self.session.get(ref_filename, timeout=self.timeout)
                response.raise_for_status()
                return Image.open(BytesIO(response.content)).convert("RGB")
            else:
                image_path = os.path.join(self.image_folder_path, ref_filename)
                jxl_path = os.path.splitext(image_path)[0] + ".jxl"
                if os.path.exists(jxl_path):
                    return color_profile_handling.open_srgb(jxl_path).convert("RGB")
                elif os.path.exists(image_path):
                    return Image.open(image_path).convert("RGB")
                else:
                    base, _ = os.path.splitext(ref_filename)
                    for ext in ("png", "jpg", "jpeg", "webp"):
                        alt = os.path.join(self.image_folder_path, f"{base}.{ext}")
                        if os.path.exists(alt):
                            return Image.open(alt).convert("RGB")
            return None
        except Exception as e:
            log.error(f"Error loading reference image '{ref_filename}' on rank {self.rank}: {e}")
            return None

    # ------------------------------------------------------------------
    # __getitem__
    # ------------------------------------------------------------------

    def __getitem__(self, index: int):
        batch = self.batches[index]

        if not self.dummy_image:
            with _cf.ThreadPoolExecutor() as executor:
                image_futures = [executor.submit(self._load_image, s) for s in batch]

                ref_futures_per_sample = []
                if self.num_reference_images is not None:
                    for sample in batch:
                        ref_list = sample.get("reference_images", [])[: self.num_reference_images]
                        ref_futures_per_sample.append(
                            [executor.submit(self._load_reference_image, r) for r in ref_list]
                        )

                raw_images = [f.result() for f in image_futures]
                raw_reference_images = (
                    [[f.result() for f in futs] for futs in ref_futures_per_sample]
                    if self.num_reference_images is not None
                    else []
                )
        else:
            raw_images = [None] * len(batch)
            raw_reference_images = [[] for _ in batch] if self.num_reference_images is not None else []

        images = []
        training_prompts = []
        loss_weighting = []
        reference_images_batch = []

        for i, sample in enumerate(batch):
            try:
                target_width, target_height = sample["bucket"]

                if not self.dummy_image:
                    img = raw_images[i]
                    if img is None:
                        continue
                    img = self.scale_and_crop_long_axis(img, target_height, target_width)
                    images.append(self.image_transforms(img))
                else:
                    images.append(torch.zeros(3, target_height, target_width))

                # Reference images
                if self.num_reference_images is not None:
                    ref_tensors = []
                    if not self.dummy_image:
                        for ref_img in (raw_reference_images[i] if i < len(raw_reference_images) else []):
                            if ref_img is not None:
                                ref_img = self.scale_and_crop_long_axis(ref_img, target_height, target_width)
                                ref_tensors.append(self.image_transforms(ref_img))
                    while len(ref_tensors) < self.num_reference_images:
                        ref_tensors.append(torch.zeros(3, target_height, target_width))
                    reference_images_batch.append(torch.stack(ref_tensors, dim=0))

                # Caption processing
                caption = sample["caption_or_tags"]

                # Unconditional dropout
                if random.random() >= 1 - self.uncond_percentage:
                    caption = ""

                if self.shuffle_tags and sample["is_tag_based"] and caption:
                    tags = caption.split(",")
                    random.shuffle(tags)
                    tags = self._sample_elements_by_percentage(
                        tags, random.uniform(1 - self.tag_drop_percentage, 1)
                    )
                    caption = ",".join(tags).lstrip()

                training_prompts.append(caption)
                loss_weighting.append(sample.get("loss_weight", 1.0))

            except Exception as e:
                log.error(f"Error processing sample '{sample['filename']}' on rank {self.rank}: {e}")
                continue

        # Empty batch fallback
        if len(images) < 1:
            log.info("Empty batch caught — fetching a random replacement batch.")
            return self.__getitem__(random.randrange(0, len(self.batches)))

        # Echo short batches
        if self.rank_batch_size > 1:
            while len(images) < self.rank_batch_size:
                log.info(f"Short batch ({len(images)}/{self.rank_batch_size}), echoing.")
                idx = random.randrange(len(images))
                images.append(images[idx])
                training_prompts.append(training_prompts[idx])
                loss_weighting.append(loss_weighting[idx])
                if self.num_reference_images is not None and reference_images_batch:
                    reference_images_batch.append(reference_images_batch[idx])

        images = torch.stack(images, dim=0)

        if self.num_reference_images is not None and reference_images_batch:
            reference_images_batch = torch.stack(reference_images_batch, dim=0)
            return images, training_prompts, index, loss_weighting, reference_images_batch

        return images, training_prompts, index, loss_weighting
