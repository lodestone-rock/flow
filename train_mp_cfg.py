import torch.multiprocessing as mp
import os
import torch

from src.trainer.train_chroma_cfg import train_chroma
if __name__ == "__main__":
    
    world_size = torch.cuda.device_count()

    # Use spawn method for starting processes
    mp.spawn(
        train_chroma,
        args=(world_size,),
        nprocs=world_size,
        join=True
    )