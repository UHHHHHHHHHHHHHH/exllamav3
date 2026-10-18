from __future__ import annotations
import atexit
import hashlib
import os
import time
import weakref
from typing import Callable
import numpy as np
import torch
from ..ext import exllamav3_ext as ext
from ..modules.quant.exl3_lib.quantize import codebook_mcg_mult, codebook_mul1_mult
from ..util.pinned_arena import PinnedArena

RECONSTRUCT_SLICE_N = 32768
REBUILD_EVERY = 8192         # counted tokens between refreshes; also the coverage window
OVERDUE_REBUILDS = 4         # refresh inline once this many are pending without the queue draining
DECAY = 0.97                 # per refresh, a half-life of ~23 refreshes (~190k tokens)
SEED_WINDOW = 16384          # prompt tokens scanned when a job starts drafting
SAVE_INTERVAL_S = 60.0
COVERAGE_ON = 0.96           # draft with the table from this windowed coverage on
COVERAGE_OFF = 0.94          # back to the full head below this


def hot_head_stats_path(model_dir: str, vocab: int) -> str:
    """Per-model counts file, keyed on the model directory and the vocabulary size, or EXL3_MTP_HOT_HEAD_STATS"""
    explicit = os.environ.get("EXL3_MTP_HOT_HEAD_STATS")
    if explicit:
        return explicit
    key = hashlib.sha1(f"{os.path.abspath(model_dir)}|{vocab}".encode()).hexdigest()[:16]
    cache = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
    return os.path.join(cache, "exllamav3", "hot_head", key + ".npz")


def _unpack_signs(bitfield: torch.Tensor) -> torch.Tensor:
    bits = bitfield.view(torch.uint16).to(torch.int)
    masks = 1 << torch.arange(16, device = bits.device)
    expanded = ((bits.unsqueeze(-1) & masks) > 0).flatten()
    return (1.0 - expanded.to(torch.half) * 2.0).contiguous()


@torch.inference_mode
def _reconstruct_head(stc, key: str, store: torch.Tensor, device: torch.device):
    vocab, hidden = store.shape
    if not stc.has_tensor(key + ".trellis"):
        store.copy_(stc.get_tensor(key + ".weight", "cpu", allow_bf16 = True)[:vocab].to(store.dtype))
        return

    trellis = stc.get_tensor(key + ".trellis", "cpu")
    def signs(name):
        if stc.has_tensor(key + "." + name + "h"):
            return stc.get_tensor(key + "." + name + "h", device)
        return _unpack_signs(stc.get_tensor(key + "." + name, device))
    suh, svh = signs("su"), signs("sv")
    mcg = stc.get_codebook_marker(key + ".mcg", codebook_mcg_mult) is not None
    mul1 = stc.get_codebook_marker(key + ".mul1", codebook_mul1_mult) is not None
    K = trellis.shape[-1] / 16
    n_total = min(trellis.shape[1] * 16, (vocab + 127) // 128 * 128)

    # One slice of packed columns at a time, so the full head never has to sit on the device
    tiles_k, words = trellis.shape[0], trellis.shape[-1]
    p_buf = torch.empty(tiles_k * (RECONSTRUCT_SLICE_N // 16) * words, dtype = torch.short, device = device)
    w_buf = torch.empty(hidden * RECONSTRUCT_SLICE_N, dtype = torch.half, device = device)
    wt_buf = torch.empty(hidden * RECONSTRUCT_SLICE_N, dtype = torch.half, device = device)
    stream = torch.cuda.current_stream(device)
    for n0 in range(0, n_total, RECONSTRUCT_SLICE_N):
        n = min(RECONSTRUCT_SLICE_N, n_total - n0)
        p = p_buf[:tiles_k * (n // 16) * words].view(tiles_k, n // 16, words)
        p.copy_(trellis[:, n0 // 16 : (n0 + n) // 16])
        w = w_buf[:hidden * n].view(hidden, n)
        ext.reconstruct_had_slice(w, p, suh, svh[n0:n0 + n], K, mcg, mul1, 0)
        wt = wt_buf[:hidden * n].view(n, hidden)
        wt.copy_(w.t())
        m = min(n0 + n, vocab) - n0
        store[n0:n0 + m].copy_(wt[:m], non_blocking = True)
        stream.synchronize()


class MTPHotHead:
    """
    Draft-only LM head over the tokens the model generates most often, held whole on the drafter's
    device. Verification still runs the target's full head, so the hot head can only change how many
    drafts are accepted, never the output.

    Membership is kept on the host at all times. Every emitted token is counted (decayed, so the table
    follows a change in workload) and checked against the current membership, which gives the coverage:
    the share of emitted tokens the table could have drafted. Each refresh (every `rebuild_every`
    counted tokens, between generations) fills the global region with the most frequent tokens and the
    lowest unseen token ids after them, and the gate turns drafting on once the coverage of the last
    window reaches COVERAGE_ON and off below COVERAGE_OFF. A smaller sequence region, recycled oldest
    first, takes each job's recent prompt tokens and every emitted token the table did not hold.

    Nothing is held on the device, and the row store is not built, until drafting is on; both are
    released again when it turns off. `load_store` builds the (vocab, hidden) host row store.
    Counts persist to `stats_path`.
    """

    def __init__(
        self,
        load_store: Callable[[], torch.Tensor],
        vocab: int,
        hidden: int,
        rows: int,
        device: torch.device | str,
        stats_path: str | None = None,
        rebuild_every: int = REBUILD_EVERY,
        dtype: torch.dtype = torch.half,
    ):
        self.load_store = load_store
        self.vocab, self.hidden, self.dtype = vocab, hidden, dtype
        self.device = torch.device(device)
        self.rows = min(rows, vocab)
        self.seq_rows = max(1, self.rows // 8)
        self.global_rows = self.rows - self.seq_rows
        self.stats_path = stats_path
        self.rebuild_every = rebuild_every
        self.debug = bool(os.environ.get("EXL3_MTP_HOT_HEAD_DEBUG"))

        self.counts = np.zeros(vocab, dtype = np.float64)
        self.checks = 0
        self.misses = 0
        self._load_stats()
        self._last_save = time.monotonic()
        self.pending = np.empty(rebuild_every, dtype = np.int64)
        self.num_pending = 0
        self.since_refresh = 0
        self.refreshes_due = 0
        self.window_hits = 0
        self.window_total = 0
        self.coverage = None

        self.slot_of = np.full(vocab, -1, dtype = np.int32)
        self.slot_ids = np.full(self.rows, -1, dtype = np.int64)
        self.seq_cursor = 0
        self.insert_slots = []
        self.insert_ids = []

        self.store = None
        self.table = self.ids = self.bias = None
        self.staging = self.insert_staging = None
        self._staging_free = None
        self._insert_free = None
        self.active = False
        self._assign_global()

        ref = weakref.ref(self)
        atexit.register(lambda: ref() is not None and ref().save(force = True))


    @classmethod
    def from_model(cls, target, device: torch.device, rows: int) -> MTPHotHead:
        """Hot head for the target's LM head; its rows are rebuilt dense from the checkpoint when drafting starts."""
        head = target.modules[target.logit_layer_idx]
        vocab, hidden = target.config.vocab_size, head.in_features
        stc, key, dev = target.config.stc, head.key, torch.device(device)

        def load_store():
            nbytes = vocab * hidden * 2
            arena = PinnedArena(nbytes, "MTP hot head row store")
            store = arena.tensor[:nbytes].view(torch.half).view(vocab, hidden)
            _reconstruct_head(stc, key, store, dev)
            torch.cuda.empty_cache()
            store._arena = arena
            return store

        return cls(load_store, vocab, hidden, rows, device, stats_path = hot_head_stats_path(target.config.directory, vocab))


    def _load_stats(self):
        if not self.stats_path or not os.path.exists(self.stats_path):
            return
        try:
            with np.load(self.stats_path) as f:
                if f["counts"].shape == self.counts.shape:
                    self.counts[:] = f["counts"]
                    self.checks = int(f["checks"])
                    self.misses = int(f["misses"])
        except (OSError, ValueError, KeyError) as e:
            print(f" !! MTP hot head stats not loaded ({self.stats_path}): {e}")


    def save(self, force: bool = False):
        if not self.stats_path:
            return
        now = time.monotonic()
        if not force and now - self._last_save < SAVE_INTERVAL_S:
            return
        self._last_save = now
        self._flush_counts()
        try:
            os.makedirs(os.path.dirname(self.stats_path) or ".", exist_ok = True)
            tmp = self.stats_path + ".tmp.npz"
            np.savez(tmp, counts = self.counts, checks = self.checks, misses = self.misses,
                     coverage = -1.0 if self.coverage is None else self.coverage, active = self.active)
            os.replace(tmp, self.stats_path)
        except OSError as e:
            print(f" !! MTP hot head stats not saved ({self.stats_path}): {e}")


    def close(self):
        self.save(force = True)
        self._deactivate()


    def observe(self, token_ids: torch.Tensor):
        """Count tokens the target emitted (a CPU tensor of IDs), score them against the table, and queue the ones it lacked."""
        ids = token_ids.view(-1).numpy()
        ids = ids[ids < self.vocab]
        n = len(ids)
        if not n:
            return
        held = self.slot_of[ids] >= 0
        self.window_hits += int(np.count_nonzero(held))
        self.window_total += n
        if not held.all():
            self._queue_seq(ids[~held])
        if self.num_pending + n > len(self.pending):
            self._flush_counts()
        self.pending[self.num_pending : self.num_pending + n] = ids
        self.num_pending += n
        self.since_refresh += n
        if self.since_refresh >= self.rebuild_every:
            self.since_refresh = 0
            self.refreshes_due += 1
            if self.refreshes_due >= OVERDUE_REBUILDS:
                self._refresh()


    def note_reject(self, token_id: int):
        """The target rejected a hot-head draft by emitting token_id; a miss if the table could not have drafted it."""
        self.checks += 1
        if token_id < self.vocab and self.slot_of[token_id] < 0:
            self.misses += 1


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


    def on_idle(self):
        """Apply a pending refresh between generations, and save on schedule."""
        if self.refreshes_due:
            self._refresh()
        self.save()


    def _flush_counts(self):
        if not self.num_pending:
            return
        np.add.at(self.counts, self.pending[:self.num_pending], 1.0)
        self.num_pending = 0


    def _refresh(self):
        self.refreshes_due = 0
        self._flush_counts()
        if self.window_total:
            self.coverage = self.window_hits / self.window_total
        self.window_hits = self.window_total = 0
        added = self._assign_global()
        self.counts *= DECAY
        was_active = self.active
        if not self.active and self.coverage is not None and self.coverage >= COVERAGE_ON:
            self._activate()
        elif self.active and self.coverage is not None and self.coverage < COVERAGE_OFF:
            self._deactivate()
        elif self.active:
            self._upload(added[0], added[1], self.staging, "_staging_free")
        if self.debug:
            print(f" -- MTP hot head: coverage {self.coverage if self.coverage is not None else 0.0:.1%} of the last window, "
                  f"{len(added[0])} rows replaced, {'drafting' if self.active else 'full head'}"
                  f"{' (switched)' if self.active != was_active else ''}", flush = True)
        if self.active != was_active:
            self.save(force = True)


    def _assign_global(self) -> tuple[np.ndarray, np.ndarray]:
        """Set the global region to the most counted tokens, then the lowest unseen ids; returns the (slots, ids) that changed."""
        n_top = min(self.global_rows, int(np.count_nonzero(self.counts)))
        want = np.argpartition(self.counts, -n_top)[-n_top:] if n_top else np.empty(0, dtype = np.int64)
        need = self.global_rows - n_top
        if need:
            free = np.ones(self.vocab, dtype = bool)
            free[want] = False
            seq_ids = self.slot_ids[self.global_rows:]
            free[seq_ids[seq_ids >= 0]] = False
            want = np.concatenate([want, np.flatnonzero(free)[:need]])
        in_want = np.zeros(self.vocab, dtype = bool)
        in_want[want] = True
        add = want[self.slot_of[want] < 0]
        g_ids = self.slot_ids[:self.global_rows]
        occupied = g_ids >= 0
        stale = np.zeros(self.global_rows, dtype = bool)
        stale[occupied] = ~in_want[g_ids[occupied]]
        slots = np.nonzero(~occupied | stale)[0][:len(add)]
        add = add[:len(slots)]
        evicted = self.slot_ids[slots]
        self.slot_of[evicted[evicted >= 0]] = -1
        self.slot_of[add] = slots
        self.slot_ids[slots] = add
        return slots.astype(np.int64), add.astype(np.int64)


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
            if self.active:
                self.insert_slots.append(slot)
                self.insert_ids.append(t)


    @torch.inference_mode
    def _activate(self):
        self.store = self.load_store()
        pin = self.device.type == "cuda"
        self.table = torch.zeros((self.rows, self.hidden), dtype = self.dtype, device = self.device)
        self.ids = torch.zeros(self.rows, dtype = torch.long, device = self.device)
        # Unfilled slots must never win the argmax
        self.bias = torch.full((1, self.rows), torch.finfo(self.dtype).min, dtype = self.dtype, device = self.device)
        self.staging = torch.empty((self.rows, self.hidden), dtype = self.dtype, pin_memory = pin)
        self.insert_staging = torch.empty((self.seq_rows, self.hidden), dtype = self.dtype, pin_memory = pin)
        held = np.flatnonzero(self.slot_ids >= 0)
        self._upload(held, self.slot_ids[held], self.staging, "_staging_free")
        self.insert_slots.clear()
        self.insert_ids.clear()
        self.active = True


    def _deactivate(self):
        self.active = False
        self.store = self.table = self.ids = self.bias = None
        self.staging = self.insert_staging = None
        self._staging_free = self._insert_free = None
        self.insert_slots.clear()
        self.insert_ids.clear()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()


    def _upload(self, slots: np.ndarray, ids: np.ndarray, staging: torch.Tensor, event_attr: str):
        if not len(ids):
            return
        free_event = getattr(self, event_attr)
        if free_event is not None:
            free_event.synchronize()
        k = len(ids)
        ids_t = torch.from_numpy(np.ascontiguousarray(ids, dtype = np.int64))
        torch.index_select(self.store, 0, ids_t, out = staging[:k])
        slots_t = torch.from_numpy(np.ascontiguousarray(slots, dtype = np.int64)).to(self.device)
        self.table.index_copy_(0, slots_t, staging[:k].to(self.device, non_blocking = True))
        self.ids.index_copy_(0, slots_t, ids_t.to(self.device))
        self.bias.index_fill_(1, slots_t, 0.0)
        if self.device.type == "cuda":
            event = torch.cuda.Event()
            event.record(torch.cuda.current_stream(self.device))
            setattr(self, event_attr, event)


    @torch.inference_mode
    def apply_inserts(self):
        if not self.active or not self.insert_slots:
            return
        # A ring wrap can queue one slot twice, and index_copy_ leaves duplicate indices undefined
        latest = dict(zip(self.insert_slots, self.insert_ids))
        self.insert_slots.clear()
        self.insert_ids.clear()
        slots = np.fromiter(latest.keys(), dtype = np.int64, count = len(latest))
        ids = np.fromiter(latest.values(), dtype = np.int64, count = len(latest))
        self._upload(slots, ids, self.insert_staging, "_insert_free")


    def argmax(self, state: torch.Tensor, params: dict) -> torch.Tensor:
        """Greedy draft token per position over the hot rows; same contract as Model.lm_head_argmax."""
        logits = torch.addmm(self.bias, state.view(-1, self.hidden), self.table.t())
        conf, idx = logits.max(dim = -1)
        out_shape = state.shape[:-1]
        if params.get("export_draft_conf"):
            params["draft_conf"] = conf.view(out_shape)
        return self.ids[idx].view(out_shape)
