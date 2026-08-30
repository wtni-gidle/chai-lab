# Copyright (c) 2024 Chai Discovery, Inc.
# Licensed under the Apache License, Version 2.0.
# See the LICENSE file for details.

"""Lightweight entity input shared by FASTA and prepared JSON readers."""

from dataclasses import dataclass


@dataclass
class Input:
    sequence: str
    entity_type: int
    entity_name: str
