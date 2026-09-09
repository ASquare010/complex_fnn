"""Replay default native CUDA blocks, including rounding, splits and delayed frees."""

from collections import defaultdict

ALIGNMENT = 512
SMALL_LIMIT = 1024 * 1024


def allocated_blocks(snapshot, device=0):
    return {
        b["address"]: b
        for s in snapshot["segments"]
        if s.get("device", device) == device
        for b in s["blocks"]
        if b["state"] == "active_allocated"
    }


class NativeBlocks:
    """Addressed events select the block; reproduce its size without redoing best-fit search."""

    def __init__(self, snapshot, device=0):
        settings = snapshot.get("allocator_settings", {})
        if settings.get("expandable_segments", False) or settings.get("max_split_size", -1) != -1:
            raise ValueError("Replay requires the audited default native splitting policy")
        if any(settings.get("roundup_power2_divisions", {}).values()):
            raise ValueError("Nondefault allocation rounding is not qualified")
        self.segments, self.blocks = {}, {}
        for segment in snapshot["segments"]:
            if segment.get("device", device) != device:
                continue
            addr = segment["address"]
            self.segments[addr] = {"size": segment["total_size"], "kind": segment["segment_type"]}
            for block in segment["blocks"]:
                key = block["address"]
                if key in self.blocks:
                    raise ValueError("Duplicate baseline allocation")
                self.blocks[key] = {
                    "addr": key,
                    "segment": addr,
                    "size": block["size"],
                    "requested_size": block.get("requested_size", 0),
                    "state": block["state"],
                    "frames": block.get("frames", []),
                    "allocated_at": -1,
                    "preexisting": True,
                }

    def live(self):
        return {k: v for k, v in self.blocks.items() if v["state"] == "active_allocated"}

    def apply(self, event, index):
        action, addr = event["action"], event.get("addr")
        if action == "segment_alloc":
            if addr in self.segments:
                raise ValueError("Duplicate segment")
            self.segments[addr] = {"size": event["size"], "kind": None}
            self.blocks[addr] = {
                "addr": addr,
                "segment": addr,
                "size": event["size"],
                "state": "inactive",
                "requested_size": 0,
            }
        elif action == "alloc":
            block = self.blocks.get(addr)
            if block is None or block["state"] != "inactive":
                raise ValueError("Allocation lacks a free block")
            requested = event["size"]
            rounded = max(ALIGNMENT, ((requested + ALIGNMENT - 1) // ALIGNMENT) * ALIGNMENT)
            segment = self.segments[block["segment"]]
            kind = "small" if rounded <= SMALL_LIMIT else "large"
            if segment["kind"] is None:
                segment["kind"] = kind
            if segment["kind"] != kind:
                raise ValueError("Allocation uses the wrong size pool")
            remainder = block["size"] - rounded
            if remainder < 0:
                raise ValueError("Requested allocation exceeds free block")
            split = remainder >= ALIGNMENT if kind == "small" else remainder > SMALL_LIMIT
            used = rounded if split else block["size"]
            if split:
                self.blocks[addr + used] = {
                    "addr": addr + used,
                    "segment": block["segment"],
                    "size": remainder,
                    "requested_size": 0,
                    "state": "inactive",
                }
            self.blocks[addr] = {
                "addr": addr,
                "segment": block["segment"],
                "size": used,
                "requested_size": requested,
                "state": "active_allocated",
                "frames": event.get("frames", []),
                "allocated_at": index,
                "preexisting": False,
            }
        elif action == "free_requested":
            block = self.blocks.get(addr)
            if block is None or block["state"] != "active_allocated":
                raise ValueError("Free refers to unknown allocation")
            if block["requested_size"] != event["size"]:
                raise ValueError("Free requested-size mismatch")
            self.blocks[addr] = {**block, "state": "active_awaiting_free"}
        elif action == "free_completed":
            block = self.blocks.get(addr)
            if block is None or block["state"] != "active_awaiting_free":
                raise ValueError("Completion without a pending free")
            segment = block["segment"]
            free = {**block, "state": "inactive", "requested_size": 0}
            self.blocks[addr] = free
            ordered = sorted((k, v) for k, v in self.blocks.items() if v["segment"] == segment)
            for left, right in zip(ordered, ordered[1:]):
                # Merge adjacent inactive pieces; the second pass below handles a three-way merge.
                la, lb = left
                ra, rb = right
                if (
                    la in self.blocks
                    and ra in self.blocks
                    and lb["state"] == rb["state"] == "inactive"
                    and la + lb["size"] == ra
                ):
                    self.blocks[la] = {**lb, "size": lb["size"] + rb["size"]}
                    del self.blocks[ra]
            ordered = sorted((k, v) for k, v in self.blocks.items() if v["segment"] == segment)
            for (la, lb), (ra, rb) in zip(ordered, ordered[1:]):
                if (
                    la in self.blocks
                    and ra in self.blocks
                    and lb["state"] == rb["state"] == "inactive"
                    and la + lb["size"] == ra
                ):
                    self.blocks[la] = {**lb, "size": lb["size"] + rb["size"]}
                    del self.blocks[ra]
        elif action == "segment_free":
            if addr not in self.segments:
                raise ValueError("Unknown segment release")
            members = [k for k, v in self.blocks.items() if v["segment"] == addr]
            if len(members) != 1 or self.blocks[members[0]]["state"] != "inactive":
                raise ValueError("Release of a live or fragmented segment")
            if self.segments[addr]["size"] != event["size"]:
                raise ValueError("Segment release size mismatch")
            del self.blocks[members[0]]
            del self.segments[addr]
        elif action not in ("snapshot", "annotate"):
            raise ValueError(f"Unsupported allocator action: {action}")


def replay_allocations(baseline, final, *, forward_end, device=0, cap=50000):
    events = final["device_traces"][device]
    prefix = baseline["device_traces"][device]
    if len(events) >= cap or events[: len(prefix)] != prefix:
        raise ValueError("Truncated or inconsistent allocator trace")
    start = len(prefix)
    if not start <= forward_end <= len(events):
        raise ValueError("Invalid phase boundary")
    allocator = NativeBlocks(baseline, device)
    current = sum(v["size"] for v in allocator.live().values())
    initial = peak = current
    peak_index = start - 1
    peak_live = list(allocator.live().values())
    points = [{"event": start - 1, "allocated_bytes": current, "phase": "baseline"}]
    for index in range(start, len(events)):
        allocator.apply(events[index], index)
        live = allocator.live()
        current = sum(v["size"] for v in live.values())
        phase = "forward" if index < forward_end else "backward"
        if events[index]["action"] in ("alloc", "free_requested"):
            points.append({"event": index, "allocated_bytes": current, "phase": phase})
        if current > peak:
            peak, peak_index, peak_live = current, index, list(live.values())
    expected = NativeBlocks(final, device)

    def key(blocks):
        return {
            k: (
                v["size"],
                v["state"],
                v["requested_size"] if v["state"] == "active_allocated" else None,
            )
            for k, v in blocks.items()
        }

    if key(allocator.blocks) != key(expected.blocks):
        raise ValueError("Replay terminal blocks differ from snapshot")
    return {
        "initial_bytes": initial,
        "final_bytes": current,
        "peak_bytes": peak,
        "peak_event": peak_index,
        "peak_phase": "baseline"
        if peak_index < start
        else "forward"
        if peak_index < forward_end
        else "backward",
        "peak_allocations": peak_live,
        "points": points,
        "events": len(events),
        "forward_end_event": forward_end,
    }


def allocation_origin(block, known, regions):
    if block["preexisting"]:
        item = known.get(str(block["addr"]))
        return {
            "category": item["category"] if item else "other_preexisting",
            "site": item["name"] if item else "unidentified before trace",
        }
    for frame in block["frames"]:
        filename = frame.get("filename", "").replace("\\", "/")
        line = frame.get("line", -1)
        for region in regions:
            if filename.endswith(region["path"]) and region["start"] <= line <= region["end"]:
                return {
                    "category": region["category"],
                    "site": f"{region['path']}:{line}",
                    "function": region["function"],
                }
    cpp = [f.get("name", "") for f in block["frames"] if "Backward" in f.get("name", "")]
    return {
        "category": "unattributed_native_backward" if cpp else "other_runtime",
        "site": cpp[0] if cpp else "no matched source region",
    }


def summarize_peak(replay, known, regions):
    categories, sites, allocations = defaultdict(int), defaultdict(int), []
    for block in replay["peak_allocations"]:
        origin = allocation_origin(block, known, regions)
        categories[origin["category"]] += block["size"]
        sites[origin["site"]] += block["size"]
        allocations.append({**block, **origin})
    assert sum(categories.values()) == replay["peak_bytes"]
    return {
        "category_bytes": dict(categories),
        "site_bytes": dict(sorted(sites.items(), key=lambda pair: -pair[1])),
        "allocations": sorted(allocations, key=lambda block: -block["size"]),
        "requested_bytes_at_allocated_peak": sum(
            b["requested_size"] for b in replay["peak_allocations"]
        ),
    }
