"""Degradation datasets and operators for FlashVSR LSWA."""

from .offline_lq import DEFAULT_LQ_OUTPUT_DIR, DEFAULT_METADATA_CSV, degrade_csv_to_lq

__all__ = ["DEFAULT_LQ_OUTPUT_DIR", "DEFAULT_METADATA_CSV", "degrade_csv_to_lq"]
