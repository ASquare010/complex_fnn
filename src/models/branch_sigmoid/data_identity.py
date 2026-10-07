"""Canonical, complete identity of decoded CPU language-training rows.

The expected identity must come from the independently parsed archive, not from
another call to the same loader. This does not load files or alter training data.
"""

import hashlib
import json
import math
import sys

import torch


PROTOCOL = "branch-data-decoded-v1"
_REQUIRED = frozenset(("ids", "score", "memory", "kind", "document_id"))
_ALLOWED = _REQUIRED | {"source_id"}


def _canonical_scalar(value):
    """Explicit tags preserve distinctions such as True, 1, 1.0 and '1'."""
    if value is None:
        return ["none"]
    if type(value) is bool:
        return ["bool", value]
    if type(value) is int:
        return ["int", str(value)]
    if type(value) is float and math.isfinite(value):
        return ["float", value.hex()]
    if type(value) is str:
        return ["str", value]
    raise ValueError(f"Unsupported row metadata type: {type(value).__name__}")


def _json_bytes(value):
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), allow_nan=False).encode("ascii")


def _frame(hasher, payload):
    hasher.update(len(payload).to_bytes(8, "little"))
    hasher.update(payload)


def _check_tensor(value, dtype, shape, label):
    if type(value) is not torch.Tensor:
        raise ValueError(f"{label} is not a plain tensor")
    if value.device.type != "cpu" or value.layout != torch.strided:
        raise ValueError(f"{label} must be a strided CPU tensor")
    if value.dtype != dtype or tuple(value.shape) != shape:
        raise ValueError(f"{label} has an invalid dtype or shape")
    if not value.is_contiguous() or value.storage_offset() != 0 or value.requires_grad:
        raise ValueError(f"{label} must be contiguous, have zero offset and require no gradients")


def decoded_fingerprint(rows):
    """Validate rows and hash every metadata value and tensor byte, in order.

    Raises ValueError for malformed rows. The protocol is a stream of frames,
    each prefixed by its unsigned eight-byte little-endian payload length:
    protocol string, JSON row count, JSON row header, then sorted field names and
    tagged JSON metadata or tensor descriptor plus raw contiguous tensor bytes.
    No sampling is used. Only train/valid callers should choose which split to load.
    """
    if sys.byteorder != "little":
        raise ValueError("This archive identity protocol requires a little-endian host")
    if type(rows) is not list or not rows:
        raise ValueError("Decoded data must be a nonempty list of rows")
    hasher = hashlib.sha256()
    _frame(hasher, PROTOCOL.encode("ascii"))
    _frame(hasher, _json_bytes(["rows", len(rows)]))
    for index, row in enumerate(rows):
        if type(row) is not dict or any(type(key) is not str for key in row):
            raise ValueError(f"Row {index} must be a dictionary with string keys")
        keys = frozenset(row)
        if not _REQUIRED <= keys or not keys <= _ALLOWED:
            raise ValueError(f"Row {index} has missing or unexpected fields")
        if type(row["kind"]) is not str or row["kind"] not in ("text", "chat"):
            raise ValueError(f"Row {index} has an invalid corpus kind")
        if type(row["document_id"]) is not str or not row["document_id"]:
            raise ValueError(f"Row {index} has an invalid document identity")
        ids = row["ids"]
        if type(ids) is not torch.Tensor or ids.ndim != 1 or ids.numel() < 2:
            raise ValueError(f"Row {index} has invalid token dimensions")
        length = ids.numel()
        _check_tensor(ids, torch.int32, (length,), f"Row {index} ids")
        _check_tensor(row["score"], torch.bool, (length,), f"Row {index} score")
        _check_tensor(
            row["memory"], torch.bfloat16, ((length - 1) // 256, 8, 256),
            f"Row {index} memory",
        )
        ordered_keys = sorted(row)
        _frame(hasher, _json_bytes(["row", index, ordered_keys]))
        for key in ordered_keys:
            value = row[key]
            _frame(hasher, _json_bytes(["field", key]))
            if key in ("ids", "score", "memory"):
                descriptor = [
                    "tensor", str(value.dtype), list(value.shape), list(value.stride()),
                    value.storage_offset(), value.requires_grad,
                ]
                _frame(hasher, _json_bytes(descriptor))
                _frame(hasher, value.contiguous().view(torch.uint8).numpy().tobytes())
            else:
                _frame(hasher, _json_bytes(_canonical_scalar(value)))
    return hasher.hexdigest()
