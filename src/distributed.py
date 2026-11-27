"""One process per GPU (NCCL); Gloo/CPU for distributed regression tests."""
from contextlib import nullcontext
from datetime import timedelta
import os
import traceback
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel


class DistributedContext:
    def __init__(self, training):
        self.world_size = int(os.environ.get("WORLD_SIZE", "1"))
        self.rank = int(os.environ.get("RANK", "0"))
        self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if self.world_size != training["num_devices"]:
            raise RuntimeError(f"Expected {training['num_devices']} processes; got {self.world_size}. "
                               "Launch using bash bash/run.sh (torchrun).")
        if training["device"] == "cuda":
            if not torch.cuda.is_available() or torch.cuda.device_count() < self.world_size:
                raise RuntimeError("Select GPU T4 x2 in Kaggle settings; two visible GPUs required")
            torch.cuda.set_device(self.local_rank)
            self.device = torch.device("cuda", self.local_rank)
            backend = "nccl"
        else:
            self.device = torch.device("cpu")
            backend = "gloo"
        self.owns_group = self.world_size > 1 and not dist.is_initialized()
        if self.owns_group:
            dist.init_process_group(backend=backend, rank=self.rank, world_size=self.world_size,
                                    timeout=timedelta(hours=2),
                                    init_method=os.environ.get("FONGBE_DDP_INIT_METHOD", "env://"))

    @property
    def primary(self):
        return self.rank == 0

    def close(self):
        if self.owns_group and dist.is_initialized():
            dist.destroy_process_group()

    def wrap(self, model):
        if self.world_size == 1:
            return model
        return DistributedDataParallel(model,
            device_ids=[self.local_rank] if self.device.type == "cuda" else None,
            find_unused_parameters=True, broadcast_buffers=False)

    def accumulation_context(self, model, synchronize):
        return model.no_sync() if self.world_size > 1 and not synchronize else nullcontext()

    def broadcast(self, value):
        if self.world_size == 1:
            return value
        values = [value if self.primary else None]
        dist.broadcast_object_list(values, src=0)
        return values[0]

    def gather(self, value):
        if self.world_size == 1:
            return [value]
        values = [None] * self.world_size
        dist.all_gather_object(values, value)
        return values

    def primary_call(self, function):
        if self.world_size == 1:
            return function()
        response = None
        if self.primary:
            try:
                response = {"value": function(), "error": None}
            except Exception:
                response = {"value": None, "error": traceback.format_exc()}
        response = self.broadcast(response)
        if response["error"]:
            raise RuntimeError("Rank-zero operation failed:\n" + response["error"])
        return response["value"]

    def any(self, value):
        t = torch.tensor(int(bool(value)), device=self.device)
        if self.world_size > 1:
            dist.all_reduce(t, op=dist.ReduceOp.MAX)
        return bool(t.item())

    def sum(self, value):
        t = torch.tensor(value, dtype=torch.float64, device=self.device)
        if self.world_size > 1:
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
        return t.item()


def optimizer_step(model, optimizer, scaler, context):
    """All ranks skip and lower AMP scale together on any nonfinite gradient."""
    scaler.unscale_(optimizer)
    invalid = torch.zeros((), dtype=torch.int32, device=context.device)
    for p in model.parameters():
        if p.grad is not None:
            invalid.bitwise_or_((~torch.isfinite(p.grad).all()).int())
    if context.any(invalid.item()):
        if not scaler.is_enabled():
            raise FloatingPointError("Nonfinite gradient on at least one rank")
        scaler.update(new_scale=scaler.get_scale() * scaler.get_backoff_factor())
        return False
    scaler.step(optimizer)
    scaler.update()
    return True
