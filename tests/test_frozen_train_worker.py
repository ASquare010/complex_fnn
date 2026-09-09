"""Protect frozen scientific jobs from changed protocols, sources and overwrites."""

import hashlib
import json

import pytest

from src.core.frozen_train_worker import read_cell, worker


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize("changed", ["protocol", "plan", "source"])
def test_changed_frozen_inputs_are_rejected(tmp_path, changed):
    plan = tmp_path / "plan.md"
    source = tmp_path / "source.py"
    protocol = tmp_path / "protocol.json"
    qualification = tmp_path / "qualification.json"
    plan.write_text("Frozen learning configuration")
    source.write_text("unchanged computation")
    raw = {"model": {"width": 384}, "training": {"seed": 29}}
    protocol.write_text(json.dumps({"plan_sha256": digest(plan), "configurations": {"cell": raw}}))
    output = tmp_path / "new_run"
    qualification.write_text(
        json.dumps(
            {
                "protocol_sha256": digest(protocol),
                "plan_path": str(plan),
                "plan_sha256": digest(plan),
                "critical_source_hashes": {str(source): digest(source)},
                "cells": {"cell": str(output)},
            }
        )
    )
    assert read_cell(protocol, qualification, "cell") == (raw, output)
    target = {"protocol": protocol, "plan": plan, "source": source}[changed]
    target.write_text(target.read_text() + "changed")
    with pytest.raises(AssertionError):
        read_cell(protocol, qualification, "cell")
    assert not output.exists()


def test_existing_trial_is_rejected_before_import_or_training(tmp_path, monkeypatch):
    output = tmp_path / "existing_run"
    output.mkdir()
    record = output / "checkpoint.txt"
    record.write_text("retained measurement")
    monkeypatch.setattr("src.core.frozen_train_worker.read_cell", lambda *args: ({}, output))
    with pytest.raises(AssertionError, match="Never overwrite"):
        worker(tmp_path / "unused", tmp_path / "unused", "cell")
    assert record.read_text() == "retained measurement"
