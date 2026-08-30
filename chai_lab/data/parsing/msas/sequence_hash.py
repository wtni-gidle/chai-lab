# Copyright (c) 2024 Chai Discovery, Inc.
# Licensed under the Apache License, Version 2.0.
# See the LICENSE file for details.

"""Stable sequence-derived names used by Chai MSA artifacts."""

import hashlib


def hash_sequence(sequence: str) -> str:
    return hashlib.sha256(sequence.encode()).hexdigest()


def expected_basename(query_sequence: str) -> str:
    seqhash = hash_sequence(query_sequence.upper())
    return f"{seqhash}.aligned.pqt"
