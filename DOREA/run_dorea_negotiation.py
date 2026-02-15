# -*- coding: utf-8 -*-
"""
run_dorea_negotiation.py
"""

from __future__ import annotations

from typing import Any

from dorea_train import run_dorea_wechat_worker


def run_training(wechat_service, stage: str = "train", results_csv: str = "custom_results.csv", *args: Any, **kwargs: Any):
    stage = (stage or "").strip().lower()
    if stage in ["train", "collect", "collect_offline", "offline"]:
        mode = "collect_offline"
    elif stage in ["online_finetune", "finetune", "online"]:
        mode = "online_finetune"
    elif stage in ["serve", "inference", "test"]:
        mode = "serve"
    else:
        mode = "collect_offline"

    # results_csv not used in this DOREA pipeline; kept for backward compatibility
    return run_dorea_wechat_worker(wechat_service, mode=mode, **kwargs)
