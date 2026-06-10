# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from omegaconf import DictConfig


def validate_typed_config(config: DictConfig, config_name: str = "default") -> DictConfig:
    del config_name
    return config
