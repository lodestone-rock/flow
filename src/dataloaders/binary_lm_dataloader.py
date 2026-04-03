"""Packed text dataloader for BinaryLM AR training.

Loads raw text from parquet files, concatenates documents with a stop token
(0xFF = 255) as separator, then slices the byte stream into fixed-length
token windows of CHARS_PER_TOKEN (64) bytes each.

This is the classical "packing" approach used in LLM pretraining:
  doc1 <0xFF> doc2 <0xFF> doc3 ...
  └──────────────────────────────┘
       sliced into T-token sequences

Each token is a 64-byte window encoded as a 512-dim {-1, +1} float vector.

Stop token (0xFF):
  - Inserted between documents as a separator
  - During inference, generation stops at the first token containing 0xFF
  - Upper 128 ASCII values (0x80-0xFF) are unused in normal ASCII text,
    so 0xFF is a safe sentinel that never appears in clean text

Encoding: {-1, +1}  (bit 1 -> +1.0, bit 0 -> -1.0)
  This is more natural for diffusion models than {0, 1} since the
  "neutral" noise prior is N(0,1) which is centred at 0, matching
  the midpoint between -1 and +1.
"""

import os
import math
import random
import logging
from typing import Optional

import torch
from torch.utils.data import Dataset, DataLoader
import pyarrow.parquet as pq
import pyarrow as pa

from src.models.binary_lm.model_ar import (
    CHARS_PER_TOKEN,
    BITS_PER_CHAR,
    BITS_PER_TOKEN,
    STOP_BYTE,
    _BIT_TABLE,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Byte stream -> token tensor
# ---------------------------------------------------------------------------

def bytes_to_tokens(byte_stream: bytes | bytearray, device: torch.device) -> torch.Tensor:
    """Convert a flat byte stream into a token tensor.

    Args:
        byte_stream: Raw bytes, length must be a multiple of CHARS_PER_TOKEN.
        device:      Target device.

    Returns:
        tokens: [T, BITS_PER_TOKEN]  float32, values in {-1, +1}
    """
    assert len(byte_stream) % CHARS_PER_TOKEN == 0, \
        f"byte_stream length {len(byte_stream)} must be multiple of {CHARS_PER_TOKEN}"

    T = len(byte_stream) // CHARS_PER_TOKEN
    byte_vals = torch.frombuffer(
        bytes(byte_stream), dtype=torch.uint8
    ).long()                                          # [T * 64]
    # Look up {-1,+1} bits: [T*64, 8] -> [T, 512]
    tokens = _BIT_TABLE[byte_vals].reshape(T, BITS_PER_TOKEN)
    return tokens.to(device)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class BinaryLMDataset(Dataset):
    """Packed text dataset for BinaryLM AR training.

    Loads all parquet files from a directory, extracts the text column,
    concatenates documents with STOP_BYTE (0xFF) separators, then slices
    the resulting byte stream into fixed-length sequences of `seq_len` tokens.

    Each item returned is a dict:
        {
            "tokens":  [seq_len, 512]  float32  {-1, +1}
            "mask":    [seq_len]       bool      always True (no padding — packed)
        }

    Parameters
    ----------
    data_dir : str
        Directory containing .parquet files (searched recursively).
    seq_len : int
        Number of tokens per training sequence.
    text_column : str
        Name of the text column in the parquet files.
    seed : int
        Random seed for shuffling documents before packing.
    rank : int
        This GPU's rank for sharding.
    num_gpus : int
        Total number of GPUs for sharding.
    max_docs : int | None
        If set, subsample this many documents (useful for quick experiments).
    """

    def __init__(
        self,
        data_dir: str,
        seq_len: int = 256,
        text_column: str = "text",
        seed: int = 42,
        rank: int = 0,
        num_gpus: int = 1,
        max_docs: Optional[int] = None,
    ):
        self.data_dir    = data_dir
        self.seq_len     = seq_len
        self.text_column = text_column
        self.seed        = seed
        self.rank        = rank
        self.num_gpus    = num_gpus
        self.max_docs    = max_docs

        self._sequences: list[bytes] = []   # list of packed byte sequences
        self._load()

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _collect_parquet_files(self) -> list[str]:
        files = []
        for root, _, fnames in os.walk(self.data_dir):
            for f in sorted(fnames):
                if f.endswith(".parquet"):
                    files.append(os.path.join(root, f))
        if not files:
            raise FileNotFoundError(f"No parquet files found under: {self.data_dir}")
        return files

    def _load(self):
        """Load parquet files, pack into byte sequences, shard for this rank."""
        rng = random.Random(self.seed)

        files = self._collect_parquet_files()
        log.info(f"[BinaryLMDataset] Found {len(files)} parquet files in {self.data_dir}")

        # Load all text documents
        all_docs: list[str] = []
        for fpath in files:
            table = pq.read_table(fpath, columns=[self.text_column])
            col   = table.column(self.text_column).to_pylist()
            all_docs.extend(d for d in col if d)

        log.info(f"[BinaryLMDataset] Loaded {len(all_docs):,} documents")

        if self.max_docs is not None:
            all_docs = rng.sample(all_docs, min(self.max_docs, len(all_docs)))
            log.info(f"[BinaryLMDataset] Subsampled to {len(all_docs):,} documents")

        # Shuffle documents
        rng.shuffle(all_docs)

        # Pack into a single byte stream with STOP_BYTE separators.
        # Non-ASCII chars are replaced with '?' (0x3F).
        #
        # Random alignment offset: prepend 0-63 PAD_BYTE (0xFE) bytes so the
        # model sees tokens that start at any byte offset within a document.
        # This matches the PAD_BYTE prepending done at inference time, keeping
        # the model in-distribution for any prefix length.
        from src.models.binary_lm.model_ar import PAD_BYTE
        pad_offset = rng.randint(0, CHARS_PER_TOKEN - 1)
        stream = bytearray([PAD_BYTE] * pad_offset)
        stop   = bytes([STOP_BYTE])
        for doc in all_docs:
            safe = doc.encode("ascii", errors="replace")
            stream += safe
            stream += stop

        total_bytes = len(stream)
        log.info(f"[BinaryLMDataset] Packed byte stream: {total_bytes:,} bytes  (align_offset={pad_offset})")

        # Align to CHARS_PER_TOKEN boundary
        seq_bytes = self.seq_len * CHARS_PER_TOKEN
        n_full    = total_bytes // seq_bytes
        stream    = stream[: n_full * seq_bytes]   # drop tail

        # Slice into sequences
        all_seqs = [
            bytes(stream[i : i + seq_bytes])
            for i in range(0, len(stream), seq_bytes)
        ]
        rng.shuffle(all_seqs)

        # Shard for this rank
        self._sequences = all_seqs[self.rank :: self.num_gpus]

        log.info(
            f"[BinaryLMDataset rank={self.rank}] "
            f"{len(self._sequences):,} sequences of {self.seq_len} tokens each"
        )

    def resample(self):
        """Reload and re-shuffle (call at end of each epoch)."""
        self.seed += 1
        self._load()

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._sequences)

    def __getitem__(self, idx: int) -> dict:
        raw = self._sequences[idx]   # bytes, length = seq_len * CHARS_PER_TOKEN

        byte_vals = torch.frombuffer(raw, dtype=torch.uint8).long()  # [seq_len * 64]
        # {-1,+1} encoding via lookup table
        tokens = _BIT_TABLE[byte_vals].reshape(self.seq_len, BITS_PER_TOKEN)  # [T, 512]
        mask   = torch.ones(self.seq_len, dtype=torch.bool)

        return {"tokens": tokens, "mask": mask}

    @staticmethod
    def collate_fn(batch: list[dict]) -> dict:
        tokens = torch.stack([b["tokens"] for b in batch])  # [B, T, 512]
        mask   = torch.stack([b["mask"]   for b in batch])  # [B, T]
        return {"tokens": tokens, "mask": mask}


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def make_dataloader(
    data_dir: str,
    seq_len: int,
    batch_size: int,
    num_workers: int = 4,
    prefetch_factor: int = 2,
    seed: int = 42,
    rank: int = 0,
    num_gpus: int = 1,
    max_docs: Optional[int] = None,
    text_column: str = "text",
) -> tuple[BinaryLMDataset, DataLoader]:
    """Create a BinaryLMDataset and DataLoader.

    Returns:
        (dataset, dataloader)  — keep dataset reference for resample() calls.
    """
    dataset = BinaryLMDataset(
        data_dir    = data_dir,
        seq_len     = seq_len,
        text_column = text_column,
        seed        = seed,
        rank        = rank,
        num_gpus    = num_gpus,
        max_docs    = max_docs,
    )
    loader = DataLoader(
        dataset,
        batch_size      = batch_size,
        shuffle         = True,
        num_workers     = num_workers,
        prefetch_factor = prefetch_factor,
        pin_memory      = True,
        collate_fn      = BinaryLMDataset.collate_fn,
        drop_last       = True,   # keep all batches the same size
    )
    return dataset, loader
