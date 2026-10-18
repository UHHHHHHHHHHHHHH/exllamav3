#pragma once

#include <torch/extension.h>
#include <tuple>

constexpr int P2P_BLOCKS = 64;
constexpr int64_t P2P_FLAGS_BYTES = 4096;

std::tuple<int64_t, py::bytes> p2p_alloc(int64_t nbytes, int64_t device);
int64_t p2p_open(py::bytes handle, int64_t device);
void p2p_free(int64_t ptr, int64_t device);
void p2p_close(int64_t ptr, int64_t device);
void p2p_all_reduce(at::Tensor data, int64_t local_base, int64_t peer_base, int64_t slot_bytes, at::Tensor epochs);
