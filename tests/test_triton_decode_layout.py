"""
Decode split-kernel row layouts and tiles (triton_paged.decode_row_layout / decode_split_config):
(query, head) rows packed densely for short multi-token queries, the wider fallback, and the
head-dim-chunked kernel for a quantized cache at head_dim 256, against an fp32 torch reference.

    python -m pytest tests/test_triton_decode_layout.py -v
"""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytest
import torch
import triton
from exllamav3.modules.attention_fn.triton_paged import (
    paged_attn_triton_decode, decode_row_layout, _paged_kv_update_kernel,
)
from exllamav3.ext import exllamav3_ext as ext
from exllamav3.constants import PAGE_SIZE

if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)

device = "cuda:0"


def ref_attn(q, k, v, window = None, softcap = 0.0, sinks = None):
    """Causal; q (B, Q, H, D) attends to k/v (B, T, KVH, D); rows are the last Q positions of T."""
    B, Q, H, D = q.shape
    T, KVH = k.shape[1], k.shape[2]
    g = H // KVH
    kk = k.repeat_interleave(g, dim = 2).float(); vv = v.repeat_interleave(g, dim = 2).float()
    s = torch.einsum("bqhd,bkhd->bhqk", q.float(), kk) * D ** -0.5
    if softcap > 0.0:
        s = torch.tanh(s / softcap) * softcap
    qpos = (T - Q + torch.arange(Q, device = q.device)).view(Q, 1)
    kpos = torch.arange(T, device = q.device).view(1, T)
    mask = kpos <= qpos
    if window is not None:
        mask &= kpos >= qpos - window
    s = s.masked_fill(~mask.view(1, 1, Q, T), -float("inf"))
    if sinks is not None:
        s = torch.cat((s, sinks.float().view(1, H, 1, 1).expand(B, H, Q, 1)), dim = -1)
        p = torch.softmax(s, -1)[..., :T]
    else:
        p = torch.softmax(s, -1)
    return torch.einsum("bhqk,bkhd->bqhd", p, vv)


def make_cache(B, T, KVH, H, D, q_len, seed):
    torch.manual_seed(seed)
    pages = -(-T // PAGE_SIZE)
    kc = torch.randn((B * pages, PAGE_SIZE, KVH, D), dtype = torch.half, device = device)
    vc = torch.randn_like(kc)
    bt = torch.randperm(B * pages, device = device, dtype = torch.int32).view(B, pages)
    sl = torch.full((B,), T - q_len, dtype = torch.int32, device = device)
    k = torch.randn((B, q_len, KVH, D), dtype = torch.half, device = device); v = torch.randn_like(k)
    q = torch.randn((B, q_len, H, D), dtype = torch.half, device = device)
    return kc, vc, bt, sl, k, v, q


def gather(kc, bt, T):
    B, pages = bt.shape
    flat = kc[bt.long().view(-1)].view(B, pages * PAGE_SIZE, kc.shape[2], kc.shape[3])
    return flat[:, :T]


def quant_cache(kc, bits):
    pages, ps, kvh, hd = kc.shape
    rows = pages * ps
    pq = torch.empty((rows, kvh * hd // 32 * bits), dtype = torch.int32, device = device)
    sc = torch.empty((rows, kvh * hd // 32), dtype = torch.half, device = device)
    ext.quant_cache_cont(kc.reshape(rows, kvh * hd).contiguous(), pq, sc, 0.0)
    deq = torch.empty((rows, kvh * hd), dtype = torch.half, device = device)
    ext.dequant_cache_cont(pq, sc, deq, 0.0)
    return pq.view(pages, ps, -1), sc.view(pages, ps, -1), deq.view(pages, ps, kvh, hd)


def rel_err(out, ref):
    return (out.float() - ref).abs().max().item() / ref.abs().max().item()


@pytest.mark.parametrize("hd", [128, 256])
@pytest.mark.parametrize("group", [1, 2, 3, 4, 6, 8, 12, 16])
@pytest.mark.parametrize("q_len", [1, 2, 3, 5, 8])
@pytest.mark.parametrize("num_splits", [None, 1, 7])
def test_decode_layout_fp16(hd, group, q_len, num_splits):
    B, kvh, T = 2, 2, 1500
    kc, vc, bt, sl, k, v, q = make_cache(B, T, kvh, kvh * group, hd, q_len, hd + group * 31 + q_len)
    out = paged_attn_triton_decode(q, k, v, kc, vc, bt, sl, causal = True, num_splits = num_splits)
    ref = ref_attn(q, gather(kc, bt, T), gather(vc, bt, T))
    err = rel_err(out, ref)
    assert err < 8e-3, f"layout {decode_row_layout(q_len, group)}: rel err {err:.3e}"


@pytest.mark.parametrize("bits", [(2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (8, 8), (8, 4), (4, 6)])
@pytest.mark.parametrize("group", [2, 6, 12])
@pytest.mark.parametrize("q_len", [1, 2, 5, 8])
@pytest.mark.parametrize("num_splits", [None, 1])
def test_decode_layout_qc256(bits, group, q_len, num_splits):
    """head_dim 256 over a packed cache: q_len > 1 runs the head-dim-chunked kernel. The reference
    uses the values the CUDA dequantizer produces, so only kernel error is measured."""
    B, kvh, T, hd = 2, 2, 1300, 256
    kbits, vbits = bits
    kc, vc, bt, sl, k, v, q = make_cache(B, T, kvh, kvh * group, hd, q_len, kbits * 10 + vbits + group + q_len)
    with torch.cuda.device(q.device):
        _paged_kv_update_kernel[(B * q_len, kvh, 1)](
            k, v, kc, vc, bt, sl, bt.shape[1], q_len, kvh, PAGE_SIZE, hd, triton.next_power_of_2(hd),
            num_warps = 2, num_stages = 3)
    qk, sk, kdeq = quant_cache(kc, kbits); qv, sv, vdeq = quant_cache(vc, vbits)
    out = paged_attn_triton_decode(q, None, None, qk, qv, bt, sl, causal = True, qc = (sk, sv, kbits, vbits),
                                   pre_appended_len = q_len, n_kv_heads_override = kvh, num_splits = num_splits)
    ref = ref_attn(q, gather(kdeq, bt, T), gather(vdeq, bt, T))
    err = rel_err(out, ref)
    assert err < 1.2e-2, f"rel err {err:.3e}"


@pytest.mark.parametrize("quant", [False, True])
@pytest.mark.parametrize("q_len", [1, 5])
@pytest.mark.parametrize("variant", ["window", "softcap", "sinks"])
@pytest.mark.parametrize("num_splits", [None, 1])
def test_decode_layout_options(quant, q_len, variant, num_splits):
    B, kvh, group, T, hd = 2, 2, 6, 1700, 256
    kc, vc, bt, sl, k, v, q = make_cache(B, T, kvh, kvh * group, hd, q_len, 1000 + q_len * 3 + len(variant))
    window = 300 if variant == "window" else None
    softcap = 20.0 if variant == "softcap" else 0.0
    sinks = torch.randn((kvh * group,), dtype = torch.float, device = device) * 2 if variant == "sinks" else None
    kw = dict(causal = True, window_size = (window, 0) if window else None, softcap = softcap, sinks = sinks,
              num_splits = num_splits)
    if quant:
        with torch.cuda.device(q.device):
            _paged_kv_update_kernel[(B * q_len, kvh, 1)](
                k, v, kc, vc, bt, sl, bt.shape[1], q_len, kvh, PAGE_SIZE, hd, triton.next_power_of_2(hd),
                num_warps = 2, num_stages = 3)
        qk, sk, kdeq = quant_cache(kc, 4); qv, sv, vdeq = quant_cache(vc, 4)
        out = paged_attn_triton_decode(q, None, None, qk, qv, bt, sl, qc = (sk, sv, 4, 4),
                                       pre_appended_len = q_len, n_kv_heads_override = kvh, **kw)
        ref = ref_attn(q, gather(kdeq, bt, T), gather(vdeq, bt, T), window, softcap, sinks)
        tol = 1.2e-2
    else:
        out = paged_attn_triton_decode(q, k, v, kc, vc, bt, sl, **kw)
        ref = ref_attn(q, gather(kc, bt, T), gather(vc, bt, T), window, softcap, sinks)
        tol = 8e-3
    err = rel_err(out, ref)
    assert err < tol, f"rel err {err:.3e}"
