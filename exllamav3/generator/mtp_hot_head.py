from __future__ import annotations
import os
import time
import numpy as np
import torch
from ..ext import exllamav3_ext as ext
from ..modules.quant.exl3_lib.quantize import codebook_mcg_mult, codebook_mul1_mult
from ..util.pinned_arena import PinnedArena

RECONSTRUCT_SLICE_N = 32768
SEED_WINDOW = 16384
SAVE_INTERVAL_S = 60.0


def _unpack_signs(bitfield: torch.Tensor) -> torch.Tensor:
    bits = bitfield.view(torch.uint16).to(torch.int)
    masks = 1 << torch.arange(16, device = bits.device)
    expanded = ((bits.unsqueeze(-1) & masks) > 0).flatten()
    return (1.0 - expanded.to(torch.half) * 2.0).contiguous()


class MTPHotHead:
    """
    Draft-only LM head over the tokens the model generates most often, held whole on the drafter's
    device. Verification still runs the target's full head, so the hot head can only change how many
    drafts are accepted, never the output.

    Every token the generator emits is counted. Once enough have been seen, the global region of the
    table holds the most frequent tokens and is refreshed every `rebuild_every` tokens by swapping in
    only the rows whose membership changed. A smaller sequence region, recycled oldest first, takes
    each job's prompt tokens and every token a draft missed, so text specific to a conversation
    becomes draftable from its next step. Counts and miss statistics persist to `stats_path`.
    """

    def __init__(
        self,
        target,
        device: torch.device,
        rows: int,
        seq_rows: int | None = None,
        stats_path: str | None = None,
        min_tokens: int = 20000,
        rebuild_every: int = 8192,
    ):
        head = target.modules[target.logit_layer_idx]
        self.vocab = target.config.vocab_size
        self.rows = min(rows, self.vocab)
        self.seq_rows = min(self.rows // 2, seq_rows if seq_rows is not None else self.rows // 8)
        self.global_rows = self.rows - self.seq_rows
        self.hidden = head.in_features
        self.device = torch.device(device)
        self.stats_path = stats_path
        self.min_tokens = min_tokens
        self.rebuild_every = rebuild_every

        row_bytes = self.hidden * 2
        self._store_arena = PinnedArena(self.vocab * row_bytes, "MTP hot head row store")
        self.store = self._store_arena.tensor[:self.vocab * row_bytes].view(torch.half).view(self.vocab, self.hidden)
        self._fill_store(target.config.stc, head.key)
        self._staging_arena = PinnedArena(self.rows * row_bytes, "MTP hot head staging")
        self.staging = self._staging_arena.tensor[:self.rows * row_bytes].view(torch.half).view(self.rows, self.hidden)
        self._staging_free = None

        self.counts = np.zeros(self.vocab, dtype = np.int64)
        self._last_save = 0.0
        self.checks = 0
        self.misses = 0
        self._load_stats()
        self.pending = np.empty(rebuild_every, dtype = np.int64)
        self.num_pending = 0

        self.table = None
        self.ids = None
        self.slot_of = np.full(self.vocab, -1, dtype = np.int32)
        self.slot_ids = np.full(self.rows, -1, dtype = np.int64)
        self.seq_cursor = 0
        self.insert_slots = []
        self.insert_ids = []
        self._insert_staging = torch.empty((self.seq_rows, self.hidden), dtype = torch.half, pin_memory = True)
        self._insert_free = None
        self.ready = False
        if self.counts.sum() >= self.min_tokens:
            self._rebuild()


    @torch.inference_mode
    def _fill_store(self, stc, key: str):
        if not stc.has_tensor(key + ".trellis"):
            w = stc.get_tensor(key + ".weight", "cpu", allow_bf16 = True)
            self.store.copy_(w[:self.vocab].to(torch.half))
            return

        dev = self.device
        trellis = stc.get_tensor(key + ".trellis", "cpu")
        def signs(name):
            if stc.has_tensor(key + "." + name + "h"):
                return stc.get_tensor(key + "." + name + "h", dev)
            return _unpack_signs(stc.get_tensor(key + "." + name, dev))
        suh, svh = signs("su"), signs("sv")
        mcg = stc.get_codebook_marker(key + ".mcg", codebook_mcg_mult) is not None
        mul1 = stc.get_codebook_marker(key + ".mul1", codebook_mul1_mult) is not None
        K = trellis.shape[-1] / 16
        n_total = min(trellis.shape[1] * 16, (self.vocab + 127) // 128 * 128)

        tiles_k, words = trellis.shape[0], trellis.shape[-1]
        p_buf = torch.empty(tiles_k * (RECONSTRUCT_SLICE_N // 16) * words, dtype = torch.short, device = dev)
        w_buf = torch.empty(self.hidden * RECONSTRUCT_SLICE_N, dtype = torch.half, device = dev)
        wt_buf = torch.empty(self.hidden * RECONSTRUCT_SLICE_N, dtype = torch.half, device = dev)
        stream = torch.cuda.current_stream(dev)
        for n0 in range(0, n_total, RECONSTRUCT_SLICE_N):
            n = min(RECONSTRUCT_SLICE_N, n_total - n0)
            p = p_buf[:tiles_k * (n // 16) * words].view(tiles_k, n // 16, words)
            p.copy_(trellis[:, n0 // 16 : (n0 + n) // 16])
            w = w_buf[:self.hidden * n].view(self.hidden, n)
            ext.reconstruct_had_slice(w, p, suh, svh[n0:n0 + n], K, mcg, mul1, 0)
            wt = wt_buf[:self.hidden * n].view(n, self.hidden)
            wt.copy_(w.t())
            m = min(n0 + n, self.vocab) - n0
            self.store[n0:n0 + m].copy_(wt[:m], non_blocking = True)
            stream.synchronize()


    def _load_stats(self):
        if not self.stats_path or not os.path.exists(self.stats_path):
            return
        with np.load(self.stats_path) as f:
            if f["counts"].shape == self.counts.shape:
                self.counts[:] = f["counts"]
                self.checks = int(f["checks"])
                self.misses = int(f["misses"])


    def _save_stats(self, force: bool = False):
        if not self.stats_path:
            return
        now = time.monotonic()
        if not force and now - self._last_save < SAVE_INTERVAL_S:
            return
        self._last_save = now
        os.makedirs(os.path.dirname(self.stats_path) or ".", exist_ok = True)
        tmp = self.stats_path + ".tmp.npz"
        np.savez(tmp, counts = self.counts, checks = self.checks, misses = self.misses)
        os.replace(tmp, self.stats_path)


    def observe(self, token_ids: torch.Tensor):
        """Count tokens the target emitted (a CPU tensor of IDs)."""
        ids = token_ids.view(-1).numpy()
        n = len(ids)
        if self.num_pending + n > self.rebuild_every:
            self._flush()
        self.pending[self.num_pending : self.num_pending + n] = ids
        self.num_pending += n


    def note_reject(self, token_id: int):
        """The target rejected a hot-head draft by emitting token_id; a miss if no draft could have hit, and then queued."""
        self.checks += 1
        if token_id < self.vocab and self.slot_of[token_id] < 0:
            self.misses += 1
            self._queue_seq(np.array([token_id], dtype = np.int64))


    def add_sequence(self, token_ids: torch.Tensor):
        """Make a job's recent prompt tokens draftable (a CPU tensor of IDs), newest first when they don't all fit."""
        tail = token_ids.view(-1).numpy()[-SEED_WINDOW:][::-1]
        tail = tail[tail < self.vocab]
        tail = tail[self.slot_of[tail] < 0]
        if len(tail) == 0:
            return
        ids, first = np.unique(tail, return_index = True)
        newest = ids[np.argsort(first)][:self.seq_rows]
        self._queue_seq(newest[::-1])


    def _queue_seq(self, ids: np.ndarray):
        for t in ids:
            if self.slot_of[t] >= 0:
                continue
            slot = self.global_rows + self.seq_cursor
            self.seq_cursor = (self.seq_cursor + 1) % self.seq_rows
            old = self.slot_ids[slot]
            if old >= 0:
                self.slot_of[old] = -1
            self.slot_of[t] = slot
            self.slot_ids[slot] = t
            self.insert_slots.append(slot)
            self.insert_ids.append(t)


    @torch.inference_mode
    def apply_inserts(self):
        """Upload queued sequence-region rows; one copy and two index writes per call, none if nothing is queued."""
        if not self.insert_slots or self.table is None:
            return
        # A ring wrap can queue one slot twice, and index_copy_ leaves duplicate indices undefined
        latest = dict(zip(self.insert_slots, self.insert_ids))
        slots = np.fromiter(latest.keys(), dtype = np.int64, count = len(latest))
        ids = np.fromiter(latest.values(), dtype = np.int64, count = len(latest))
        self.insert_slots.clear()
        self.insert_ids.clear()
        if self._insert_free is not None:
            self._insert_free.synchronize()
        k = len(ids)
        ids_t = torch.from_numpy(ids)
        torch.index_select(self.store, 0, ids_t, out = self._insert_staging[:k])
        slots_t = torch.from_numpy(slots).to(self.device)
        self.table.index_copy_(0, slots_t, self._insert_staging[:k].to(self.device, non_blocking = True))
        self.ids.index_copy_(0, slots_t, ids_t.to(self.device))
        self.bias.index_fill_(1, slots_t, 0.0)
        self._insert_free = torch.cuda.Event()
        self._insert_free.record(torch.cuda.current_stream(self.device))


    def _flush(self):
        p = self.pending[:self.num_pending]
        np.add.at(self.counts, p[p < self.vocab], 1)
        self.num_pending = 0
        was_ready = self.ready
        if self.counts.sum() >= max(1, self.min_tokens):
            self._rebuild()
        self._save_stats(force = self.ready and not was_ready)


    @torch.inference_mode
    def _rebuild(self):
        seen = int(np.count_nonzero(self.counts))
        n_top = min(self.global_rows, seen)
        top = np.argpartition(self.counts, -n_top)[-n_top:]
        add = top[self.slot_of[top] < 0]
        if self.table is None:
            self.table = torch.zeros((self.rows, self.hidden), dtype = torch.half, device = self.device)
            self.ids = torch.zeros(self.rows, dtype = torch.long, device = self.device)
            # Unfilled slots must never win the argmax
            self.bias = torch.full((1, self.rows), torch.finfo(torch.half).min, dtype = torch.half, device = self.device)
        if len(add) == 0:
            self.ready = True
            return

        in_top = np.zeros(self.vocab, dtype = bool)
        in_top[top] = True
        g_ids = self.slot_ids[:self.global_rows]
        occupied = g_ids >= 0
        stale = np.zeros(self.global_rows, dtype = bool)
        stale[occupied] = ~in_top[g_ids[occupied]]
        slots = np.nonzero(~occupied | stale)[0][:len(add)]
        add = add[:len(slots)]

        self.slot_of[self.slot_ids[slots][self.slot_ids[slots] >= 0]] = -1
        self.slot_of[add] = slots
        self.slot_ids[slots] = add

        if self._staging_free is not None:
            self._staging_free.synchronize()
        k = len(add)
        add_t = torch.from_numpy(add)
        torch.index_select(self.store, 0, add_t, out = self.staging[:k])
        rows = self.staging[:k].to(self.device, non_blocking = True)
        slots_t = torch.from_numpy(slots.astype(np.int64)).to(self.device)
        self.table.index_copy_(0, slots_t, rows)
        self.ids.index_copy_(0, slots_t, add_t.to(self.device))
        self.bias.index_fill_(1, slots_t, 0.0)
        self._staging_free = torch.cuda.Event()
        self._staging_free.record(torch.cuda.current_stream(self.device))
        self.ready = True


    def argmax(self, state: torch.Tensor, params: dict) -> torch.Tensor:
        """Greedy draft token per position over the hot rows; same contract as Model.lm_head_argmax."""
        logits = torch.addmm(self.bias, state.view(-1, self.hidden), self.table.t())
        conf, idx = logits.max(dim = -1)
        out_shape = state.shape[:-1]
        if params.get("export_draft_conf"):
            params["draft_conf"] = conf.view(out_shape)
        return self.ids[idx].view(out_shape)
