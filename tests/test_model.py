import copy
import torch
from src.common import load_config
from src.dataset import duration_steps, output_frames
from src.model import ZipformerCTC, ctc_losses, output_lengths
from src.vendor.icefall.optim import ScaledAdam
from src.vendor.icefall import torch_swoosh


def small_config():
    c = copy.deepcopy(load_config("data/configs/training/proposal01.yaml"))
    c["model"].update(encoder_dim=[32,32], encoder_unmasked_dim=[32,32],
                      num_encoder_layers=[1,1], downsampling_factor=[1,2],
                      feedforward_dim=[64,64], num_heads=[4,4],
                      cnn_module_kernel=[7,7], query_head_dim=4, value_head_dim=4,
                      pos_head_dim=2, pos_dim=16)
    return c


def test_swoosh_derivative():
    for f, both in [(torch_swoosh.swoosh_l, torch_swoosh.swoosh_l_forward_and_deriv),
                    (torch_swoosh.swoosh_r, torch_swoosh.swoosh_r_forward_and_deriv)]:
        x = torch.linspace(-50,50,101,requires_grad=True)
        f(x).sum().backward()
        assert torch.isfinite(f(x)).all()
        assert torch.allclose(x.grad,both(x.detach())[1],atol=1e-6)


def test_full_small_model_forward_backward():
    torch.set_num_threads(2)
    cfg = load_config("data/configs/training/proposal01.yaml")
    m = ZipformerCTC(cfg["model"],37)
    assert 21_000_000 < sum(p.numel() for p in m.parameters()) < 24_000_000
    m.train(); m.set_step(100)
    p,l = m(torch.randn(2,101,80),torch.tensor([101,91]))
    batch = {"labels":torch.tensor([1,2,3,1,1]),"label_lengths":torch.tensor([3,2]),
             "minimum_lengths":torch.tensor([3,3])}
    loss=ctc_losses(p,l,batch,36).mean()
    assert torch.isfinite(loss)
    loss.backward()
    opt=ScaledAdam(m.named_parameters(),lr=.04,clipping_scale=2.0)
    opt.step()
    assert all(torch.isfinite(v).all() for v in m.parameters())
    m.eval()
    with torch.inference_mode():
        p,l=m(torch.randn(1,117,80),torch.tensor([117]))
    assert p.shape[:2] == (1,28)
    assert l.item()==28


def test_host_output_lengths_match_model():
    cfg = small_config()
    m = ZipformerCTC(cfg["model"], 37).eval()
    lens = torch.tensor([101, 117, 250, 33])
    with torch.inference_mode():
        _, model_lens = m(torch.randn(4, 250, 80), lens)
    assert torch.equal(model_lens.cpu(), output_lengths(lens))
    with torch.inference_mode():
        for t in range(20, 3001, 97):
            assert output_lengths(torch.tensor([t])).item() == m(torch.randn(1, t, 80), torch.tensor([t]))[1].item()


def test_transient_module_state_round_trip():
    from src.model import transient_state, load_transient_state
    cfg = small_config()
    a, b = ZipformerCTC(cfg["model"], 37), ZipformerCTC(cfg["model"], 37)
    state = transient_state(a)
    assert any(k.endswith(".prob") for k in state) and any(k.endswith(".mem_cutoff") for k in state)
    for name, m in a.named_modules():
        if hasattr(m, "min_prob"):
            m.prob = m.min_prob
        if hasattr(m, "mem_cutoff"):
            m.mem_cutoff(12345.0)
    load_transient_state(b, transient_state(a))
    assert transient_state(b) == transient_state(a) != state


def test_set_step_updates_all_scheduled_modules():
    m = ZipformerCTC(small_config()["model"], 37)
    m.set_step(123)
    m.set_step(4567)
    counts = {mod.batch_count for mod in m.modules() if hasattr(mod, "batch_count")}
    assert counts == {4567.0}


def _check_steps(rows, steps, max_seconds, max_utts, world_size):
    flat = [i for step in steps for mb in step for i in mb]
    assert sorted(flat) == list(range(len(rows)))  # every row exactly once
    for step in steps:
        assert len(step) == world_size and all(step)  # one non-empty microbatch per rank
        for mb in step:
            assert len(mb) <= max_utts
            assert max(float(rows[i]["duration"]) for i in mb) * len(mb) <= max_seconds + 1e-9


def test_duration_sampler_budget_coverage_reproducibility():
    rows = [{"duration": (i * 7919) % 290 / 10 + 0.5} for i in range(1003)]
    for ws in (1, 2):
        a = duration_steps(rows, 45.0, 16, 128, 42, 0, ws)
        _check_steps(rows, a, 45.0, 16, ws)
        assert a == duration_steps(rows, 45.0, 16, 128, 42, 0, ws)
        assert a != duration_steps(rows, 45.0, 16, 128, 42, 1, ws)


def test_duration_sampler_on_real_train_split():
    from src.common import read_csv
    rows = read_csv("data/splits/all_labels_valspk_jdg_v1/train.csv")
    steps = duration_steps(rows, 450, 256, 4096, 42, 0, 2)
    _check_steps(rows, steps, 450, 256, 2)
    padded = sum(max(float(rows[i]["duration"]) for i in mb) * len(mb) for s in steps for mb in s)
    audio = sum(float(r["duration"]) for r in rows)
    assert padded / audio < 1.05  # bucketing keeps padding overhead small
    assert 1200 < len(steps) < 1600  # ~900 s nominal audio per optimizer step


def test_cuda_amp_when_available():
    if not torch.cuda.is_available():
        import pytest
        pytest.skip("No local CUDA; exercised by bash/setup.sh on Kaggle")
    m=ZipformerCTC(small_config()["model"],37).cuda().train()
    m.set_step(0)
    with torch.autocast("cuda",dtype=torch.float16):
        p,l=m(torch.randn(2,101,80,device="cuda"),torch.tensor([101,91],device="cuda"))
    loss=ctc_losses(p,l,{"labels":torch.tensor([1,2,1,2]),"label_lengths":torch.tensor([2,2]),
                        "minimum_lengths":torch.tensor([2,2])},36).mean()
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters())
