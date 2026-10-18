"""
MTPHotHead bookkeeping on CPU with a synthetic row store: nothing is built or allocated until the
coverage gate turns drafting on, and everything is released when it turns off; the table is full
from the start; it only ever answers with tokens it holds, picks the same token as the dense head
whenever that token is held, never lets an unfilled slot win, keeps one slot per token through
refreshes and ring wraps, and round-trips its counts.
"""

import numpy as np
import pytest
import torch

from exllamav3.generator.mtp_hot_head import MTPHotHead

pytestmark = pytest.mark.nogpu

VOCAB, HIDDEN, ROWS = 512, 16, 64


class Store:
    def __init__(self, seed = 0):
        self.rows = torch.randn((VOCAB, HIDDEN), generator = torch.Generator().manual_seed(seed))
        self.loads = 0

    def __call__(self):
        self.loads += 1
        return self.rows


def make_head(tmp_path = None, store = None):
    store = store or Store()
    path = str(tmp_path / "stats.npz") if tmp_path is not None else None
    head = MTPHotHead(store, VOCAB, HIDDEN, ROWS, "cpu", stats_path = path, rebuild_every = 64, dtype = torch.float32)
    return head, store


def feed(head, ids, repeat = 1):
    for _ in range(repeat):
        for t in ids:
            head.observe(torch.tensor([t]))
    head.on_idle()


def held_tokens(head):
    return head.slot_ids[head.slot_ids >= 0]


def assert_consistent(head):
    held = np.flatnonzero(head.slot_ids >= 0)
    assert len(set(head.slot_ids[held].tolist())) == len(held)                # one slot per token
    assert all(head.slot_of[head.slot_ids[s]] == s for s in held)
    assert int((head.slot_of >= 0).sum()) == len(held)
    if head.active:
        idx = torch.from_numpy(held)
        assert torch.equal(head.ids[idx], torch.from_numpy(head.slot_ids[held]))
        assert torch.equal(head.table[idx], head.store[head.slot_ids[held]])


def test_inactive_head_builds_and_holds_nothing():
    head, store = make_head()
    feed(head, list(range(300, 500)))                        # mostly outside the table: low coverage
    assert not head.active
    assert store.loads == 0
    assert head.store is None and head.table is None


def test_table_is_full_from_the_start():
    head, _ = make_head()
    assert int((head.slot_ids[:head.global_rows] >= 0).sum()) == head.global_rows
    assert set(held_tokens(head).tolist()) == set(range(head.global_rows))   # the lowest ids before any count


def test_gate_turns_on_with_coverage_and_off_without():
    head, store = make_head()
    feed(head, list(range(20)), repeat = 10)                 # all held
    assert head.active and store.loads == 1
    assert_consistent(head)
    feed(head, list(range(100, 500)))                        # 400 distinct tokens the table cannot hold in time
    assert not head.active
    assert head.store is None and head.table is None


def test_argmax_matches_dense_when_held_and_answers_only_held_tokens():
    head, _ = make_head()
    feed(head, list(range(20)), repeat = 10)
    held = set(held_tokens(head).tolist())
    state = torch.randn((32, 1, HIDDEN), generator = torch.Generator().manual_seed(1))
    params = {"export_draft_conf": True}
    ids = head.argmax(state, params).view(-1)
    dense = (state.view(-1, HIDDEN) @ head.store.t()).argmax(-1)
    for i, d in zip(ids.tolist(), dense.tolist()):
        assert i in held
        if d in held:
            assert i == d
    assert params["draft_conf"].shape == (32, 1)


def test_unfilled_slots_never_win():
    store = Store()
    store.rows[:, 0] = store.rows[:, 0].abs() + 1.0
    head, _ = make_head(store = store)
    feed(head, [5] * 200)
    assert head.active and (head.slot_ids[head.global_rows:] < 0).any()      # sequence region still empty
    state = torch.zeros((1, 1, HIDDEN))
    state[..., 0] = -1.0                                                       # every held logit is negative
    held = held_tokens(head)
    expected = held[(store.rows[held] @ state.view(-1)).argmax()].item()
    assert head.argmax(state, {}).item() == expected


def test_emitted_tokens_outside_the_table_become_draftable_and_the_ring_wraps():
    head, _ = make_head()
    feed(head, list(range(20)), repeat = 10)
    first = list(range(300, 300 + head.seq_rows))
    last = list(range(400, 400 + head.seq_rows))
    for t in first + list(range(350, 350 + head.seq_rows)) + last:
        head.observe(torch.tensor([t]))
    head.apply_inserts()
    assert all(head.slot_of[t] >= 0 for t in last)
    assert all(head.slot_of[t] < 0 for t in first)
    assert_consistent(head)


def test_prompt_seeding_prefers_newest_and_skips_out_of_vocab():
    head, _ = make_head()
    feed(head, list(range(20)), repeat = 10)
    prompt = torch.tensor([list(range(100, 100 + 2 * head.seq_rows)) + [VOCAB + 7]])
    head.add_sequence(prompt)
    head.apply_inserts()
    assert all(head.slot_of[t] >= 0 for t in range(100 + head.seq_rows, 100 + 2 * head.seq_rows))
    assert_consistent(head)


def test_stats_round_trip_and_decay(tmp_path):
    head, _ = make_head(tmp_path)
    feed(head, list(range(10)), repeat = 20)
    head.note_reject(400)
    head.save(force = True)
    counts = head.counts.copy()
    again, _ = make_head(tmp_path)
    assert np.array_equal(again.counts, counts)
    assert again.checks == 1 and again.misses == 1
    before = again.counts.sum()
    feed(again, [1] * 70)
    assert again.counts.sum() < before + 70                   # the refresh decayed the counts
