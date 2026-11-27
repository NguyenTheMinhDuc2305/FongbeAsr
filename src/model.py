"""Actual icefall Zipformer2 encoder plus a small character CTC head."""
import torch
from torch import nn
from src.vendor.icefall.zipformer import Zipformer2
from src.vendor.icefall.subsampling import Conv2dSubsampling
from src.vendor.icefall.scaling import ScheduledFloat


class ZipformerCTC(nn.Module):
    def __init__(self, cfg, vocab_size, feature_dim=80):
        super().__init__()
        self.config = cfg
        if cfg["initialization"] != "scratch":
            raise ValueError("P01 supports scratch initialization; resume via checkpoint")
        dropout = lambda: ScheduledFloat((0.0, cfg["dropout_start"]),
                                         (float(cfg["dropout_end_step"]), cfg["dropout_end"]),
                                         default=cfg["dropout_end"])
        self.encoder_embed = Conv2dSubsampling(feature_dim, cfg["encoder_dim"][0], dropout=dropout())
        keys = ["encoder_dim", "encoder_unmasked_dim", "num_encoder_layers", "downsampling_factor",
                "feedforward_dim", "num_heads", "query_head_dim", "value_head_dim", "pos_head_dim",
                "pos_dim", "cnn_module_kernel", "output_downsampling_factor", "causal", "warmup_batches"]
        args = {k: tuple(cfg[k]) if isinstance(cfg[k], list) else cfg[k] for k in keys}
        self.encoder = Zipformer2(**args, dropout=dropout(), chunk_size=(-1,), left_context_frames=(-1,))
        self.ctc_output = nn.Sequential(nn.Dropout(cfg["ctc_dropout"]),
                                        nn.Linear(max(cfg["encoder_dim"]), vocab_size))

    def set_step(self, step):
        for name, module in self.named_modules():
            if hasattr(module, "batch_count"):
                module.batch_count = float(step)
            if hasattr(module, "name"):
                module.name = name

    def forward(self, features, lengths):
        x, lengths = self.encoder_embed(features, lengths)
        mask = torch.arange(x.size(1), device=x.device)[None, :] >= lengths[:, None]
        x, lengths = self.encoder(x.transpose(0, 1), lengths, mask)
        return self.ctc_output(x.transpose(0, 1)).float().log_softmax(-1), lengths


def ctc_losses(log_probs, lengths, batch, blank_id):
    feasible = lengths.cpu() >= batch["minimum_lengths"]
    if not bool(feasible.all()):
        raise ValueError("CTC-infeasible targets escaped preflight")
    # FP32 CTC, no zero_infinity hiding invalid samples.
    return torch.nn.functional.ctc_loss(
        log_probs.transpose(0, 1).contiguous(), batch["labels"].cpu(),
        lengths.cpu(), batch["label_lengths"].cpu(), blank=blank_id,
        reduction="none", zero_infinity=False,
    ) / batch["label_lengths"].to(log_probs.device).clamp_min(1)
