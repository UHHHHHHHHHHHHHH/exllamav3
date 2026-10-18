"""
MTPHotHead bookkeeping on CPU with a synthetic row store: the table only ever answers with tokens it
holds, picks the same token as the dense head whenever that token is held, never lets an unfilled
slot win, keeps one slot per token through refreshes and ring wraps, and round-trips its counts.
"""

import numpy as np
import pytest
import torch

from exllamav3.generator.mtp_hot_head import MTPHotHead

pytestmark = pytest.mark.nogpu

VOCAB, HIDDEN, ROWS = 512, 16, 64


def make_head(tmp_path = None, min_tokens = 100, rebuild_every = 64, seed = 0):
    g = torch.Generator().manual_seed(seed)
    store = torch.randn((VOCAB, HIDDEN), generator = g)
    path = str(tmp_path / "stats.npz") if tmp_path is not None else None
    return MTPHotHead(store, ROWS, "cpu", stats_path = path, min_tokens = min_tokens, rebuild_every = rebuild_every)


def feed(head, ids, repeat = 1):
    for _ in range(repeat):
        for t in ids:
            head.observe(torch.tensor([t]))
    head.on_idle()


def assert_consistent(head):
    held = head.slot_ids >= 0
    assert len(set(head.slot_ids[held].tolist())) == int(held.sum())          # one slot per token
    for slot in np.nonzero(held)[0]:
        assert head.slot_of[head.slot_ids[slot]] == slot
    assert int((head.slot_of >= 0).sum()) == int(held.sum())
    assert torch.equal(head.ids[torch.from_numpy(np.nonzero(held)[0])], torch.from_numpy(head.slot_ids[held]))
    assert torch.equal(head.table[torch.from_numpy(np.nonzero(held)[0])], head.store[head.slot_ids[held]])


def test_not_ready_until_min_tokens():
    head = make_head(min_tokens = 1000)
    feed(head, range(50))
    assert not head.ready


def test_global_region_holds_most_frequent():
    head = make_head()
    feed(head, list(range(10)), repeat = 20)
    feed(head, list(range(10, 400)))
    assert head.ready
    assert all(head.slot_of[t] >= 0 for t in range(10))
    assert int((head.slot_ids[:head.global_rows] >= 0).sum()) <= head.global_rows
    assert_consistent(head)


def test_argmax_matches_dense_when_held_and_answers_only_held_tokens():
    head = make_head()
    feed(head, list(range(VOCAB)), repeat = 2)
    held = set(head.slot_ids[head.slot_ids >= 0].tolist())
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
    head = make_head()
    feed(head, [5] * 200)
    assert head.ready and int((head.slot_ids >= 0).sum()) == 1
    state = -head.store[5].view(1, 1, HIDDEN)                  # every real logit is negative
    assert head.argmax(state, {}).item() == 5


def test_misses_become_draftable_and_ring_wraps():
    head = make_head()
    feed(head, list(range(20)), repeat = 10)
    for t in range(300, 300 + 3 * head.seq_rows):            # wraps the ring three times
        head.note_reject(t)
    head.apply_inserts()
    assert head.misses == 3 * head.seq_rows
    recent = range(300 + 2 * head.seq_rows, 300 + 3 * head.seq_rows)
    assert all(head.slot_of[t] >= 0 for t in recent)
    assert all(head.slot_of[t] < 0 for t in range(300, 300 + head.seq_rows))
    assert_consistent(head)


def test_prompt_seeding_prefers_newest_and_skips_out_of_vocab():
    head = make_head()
    feed(head, list(range(20)), repeat = 10)
    prompt = torch.tensor([list(range(100, 100 + 2 * head.seq_rows)) + [VOCAB + 7]])
    head.add_sequence(prompt)
    head.apply_inserts()
    newest = range(100 + head.seq_rows, 100 + 2 * head.seq_rows)
    assert all(head.slot_of[t] >= 0 for t in newest)
    assert_consistent(head)


def test_refresh_keeps_slots_unique_after_sequence_inserts():
    head = make_head()
    feed(head, list(range(30)), repeat = 10)
    for t in range(200, 210):
        head.note_reject(t)
    head.apply_inserts()
    feed(head, list(range(200, 210)), repeat = 50)            # now globally frequent too
    assert_consistent(head)


def test_stats_round_trip_and_decay(tmp_path):
    head = make_head(tmp_path)
    feed(head, list(range(10)), repeat = 20)
    head.note_reject(400)
    head.save(force = True)
    counts = head.counts.copy()
    again = make_head(tmp_path)
    assert np.array_equal(again.counts, counts)
    assert again.checks == 1 and again.misses == 1
    assert again.ready
    before = again.counts.sum()
    feed(again, [1] * 70)
    assert again.counts.sum() < before + 70                   # the refresh decayed the counts
