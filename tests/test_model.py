"""Focused checks for correctness, boundaries and expected trends."""

import math

import pytest

from npu_explorer.model import Architecture, Hardware, Workload, choose_tile_size, dram_transfer_bytes, estimate


def setup():
    w = Workload(m=8, n=8, k=8, bytes_per_element=1)
    h = Hardware(macs_per_second=1e9, dram_bandwidth_gbps=1, dram_energy_pj_per_byte=100, mac_energy_pj=1)
    a = Architecture("Test", 1, 100, 2, 1, 0, 1)
    return w, h, a


def test_gemm_mac_count():
    assert Workload(3, 4, 5).macs == 60


def test_exact_dram_transfers_single_tile():
    assert dram_transfer_bytes(Workload(2, 2, 2), 2) == 24
    assert dram_transfer_bytes(Workload(2, 2, 2), 1) == 32


def test_uneven_dimensions():
    assert dram_transfer_bytes(Workload(3, 2, 5), 2) == 59  # A/B=35 bytes plus six 4-byte outputs


def test_tile_size_obeys_capacity():
    w = Workload(128, 128, 128, bytes_per_element=2)
    assert choose_tile_size(w, (2 * 2 + 4) * 16 * 16) == 16
    assert choose_tile_size(w, (2 * 2 + 4) * 16 * 16 - 1) == 8


def test_larger_tile_reduces_or_holds_dram_traffic():
    w = Workload(256, 256, 256)
    assert dram_transfer_bytes(w, 128) < dram_transfer_bytes(w, 32)


def test_latency_is_roofline_maximum():
    w, h, a = setup()
    outcome = estimate(w, h, a)
    expected = max(outcome["compute_latency_ms"], outcome["dram_latency_ms"], outcome["onchip_latency_ms"])
    assert math.isclose(outcome["estimated_latency_ms"], expected)


def test_nonnegative_energy_area():
    w, h, a = setup()
    outcome = estimate(w, h, a)
    assert outcome["estimated_energy_mj"] > 0
    assert outcome["memory_area_mm2"] >= 0


def test_invalid_parameters_rejected():
    with pytest.raises(ValueError):
        Workload(0, 8, 8)
    with pytest.raises(ValueError):
        choose_tile_size(Workload(2, 2, 2), 2)
