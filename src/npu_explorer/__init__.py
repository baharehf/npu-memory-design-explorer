"""Small, transparent analytical simulator for educational NPU research."""

from .model import Architecture, Hardware, Workload, estimate, choose_tile_size

__all__ = ["Architecture", "Hardware", "Workload", "estimate", "choose_tile_size"]
