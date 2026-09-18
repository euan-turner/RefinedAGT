"""
Smoke test for NCCL on one Isambard-AI node: the MIN/MAX all-reduces the sharded refinement uses.

Usage: torchrun --standalone --nproc-per-node 4 nccl_check.py
Key external dependencies: torch.distributed (NCCL).
"""

import os

import torch
import torch.distributed as dist

local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)
dist.init_process_group("nccl", device_id=torch.device(f"cuda:{local_rank}"))
rank, world_size = dist.get_rank(), dist.get_world_size()
lower = torch.full((1 << 20,), float(rank), device="cuda")
upper = lower.clone()
dist.all_reduce(lower, op=dist.ReduceOp.MIN)
dist.all_reduce(upper, op=dist.ReduceOp.MAX)
assert lower.eq(0).all() and upper.eq(world_size - 1).all(), "all-reduce mismatch"
if rank == 0:
    print(f"NCCL MIN/MAX all-reduce OK on {world_size} ranks")
dist.destroy_process_group()
