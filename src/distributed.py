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


def assert_ranks_synchronized(model, context, where):
    """Fail fast if DDP replicas diverge (per-tensor float64 sums must match bitwise)."""
    if context.world_size == 1:
        return
    with torch.no_grad():
        digest = torch.stack([p.detach().double().sum() for p in model.parameters()]).cpu()
    digests = context.gather(digest)
    bad = [r for r, d in enumerate(digests) if not torch.equal(digests[0], d)]
    if bad:
        names = [n for n, _ in model.named_parameters()]
        first = int((digests[0] != digests[bad[0]]).nonzero()[0])
        raise RuntimeError(f"DDP replicas diverged at {where}: ranks {bad} differ from rank 0, "
                           f"first tensor {names[first]}")


def optimizer_step(model, optimizer, scaler, context):
    """One host sync per update. If ANY rank has a nonfinite gradient, every rank skips and
    GradScaler backs off normally (growth tracker reset), keeping replicas identical."""
    scaler.unscale_(optimizer)
    if scaler.is_enabled():
        # unscale_ already ran a fused nonfinite check; reuse it instead of per-tensor kernels.
        found = scaler._per_optimizer_states[id(optimizer)]["found_inf_per_device"]
        local = sum(float(v.item()) for v in found.values())
    else:
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        local = 0.0 if not grads else float(
            (~torch.isfinite(torch.stack(torch._foreach_norm(grads)))).any().item())
    if context.any(local > 0):
        if not scaler.is_enabled():
            raise FloatingPointError("Nonfinite gradient on at least one rank")
        for v in found.values():
            v.fill_(1.0)
        scaler.step(optimizer)  # skipped because found_inf > 0
        scaler.update()
        return False
    scaler.step(optimizer)
    scaler.update()
    return True
