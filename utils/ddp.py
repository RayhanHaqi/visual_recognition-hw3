import os

import torch
import torch.distributed as dist


def is_dist_available():
    return dist.is_available() and dist.is_initialized()


def get_rank():
    return dist.get_rank() if is_dist_available() else 0


def get_world_size():
    return dist.get_world_size() if is_dist_available() else 1


def is_main():
    return get_rank() == 0


def init_distributed():
    if "LOCAL_RANK" not in os.environ:
        return False, 0, 1, 0
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method="env://")
    dist.barrier()
    return True, rank, world_size, local_rank


def cleanup_distributed():
    if is_dist_available():
        dist.destroy_process_group()
