from pathlib import Path

R = Path("results/interleaved_training_v1")
old = Path("results/paired_complete_training_v1/loop.py").read_text()
new = old.replace(
    '    label = fixture["label"] + "__" + mode',
    '    setup_started = time.perf_counter()\n    label = fixture["label"] + "__" + mode',
)
new = new.replace('seed=101, precision="fp32"', 'seed=fixture["seed"], precision="fp32"')
new = new.replace("Transformer(cfg, 101)", 'Transformer(cfg, fixture["seed"])')
new = new.replace('"cuda", 10101)', '"cuda", fixture["seed"] + 10000)')
new = new.replace("cfg, 801, 101)", 'cfg, 801, fixture["seed"])')
new = new.replace("cfg, 830, 101)", 'cfg, 830, fixture["seed"])')
new = new.replace(
    "    del source, state, group",
    "    del source, state, group\n    setup_wall_ms = 1000 * (time.perf_counter() - setup_started)",
)
new = new.replace(
    "        label=label,\n        fixture=fixture,",
    "        label=label,\n        setup_wall_ms=setup_wall_ms,\n        fixture=fixture,",
)
(R / "loop.py").write_text(new)
w = Path("results/partial_offload_training_v1/worker.py").read_text()
w = w[: w.index("\ndef run():")]
w = w.replace(
    "from results.paired_complete_training_v1 import loop",
    "from results.interleaved_training_v1 import loop",
)
w = w.replace(
    'ROOT = Path("results/partial_offload_training_v1")',
    'ROOT = Path("results/interleaved_training_v1")',
)
w = w.replace("import sys\n", "")
w += """
def run():
    p = read(ROOT / "protocol.json")
    for field in ("sources", "input_hashes", "maintained_files"):
        hashes(p[field])
    assert os.environ.get("CUBLAS_WORKSPACE_CONFIG") is None
    torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False
    assert torch.backends.cuda.cublas_workspace_size() == 8.125 * 2**20
    settings = dict(environment=environment(), deterministic=False,
                    workspace_bytes=torch.backends.cuda.cublas_workspace_size(),
                    threads=torch.get_num_threads(), tf32=torch.backends.cuda.matmul.allow_tf32)
    boundaries = [boundary()]
    fields = "timestamp,index,uuid,pstate,temperature.gpu,clocks.sm,clocks.mem,power.draw,power.limit,utilization.gpu"
    metadata = dict(start_ns=time.time_ns(), timezone=str(datetime.datetime.now().astimezone().tzinfo), fields=fields)
    with (ROOT / "telemetry.csv").open("x") as out, (ROOT / "telemetry.stderr").open("x") as err:
        monitor = subprocess.Popen(["nvidia-smi", "--query-gpu="+fields, "--format=csv,noheader,nounits", "-lms", "200"], stdout=out, stderr=err, creationflags=subprocess.CREATE_NO_WINDOW)
        try:
            for row in p["schedule"]:
                folder = ROOT / f"case{row['index']:02d}"
                assert not (folder / "case.json").exists()
                start = time.perf_counter()
                result = one(row, p, folder)
                segment_wall_ms = 1000 * (time.perf_counter() - start)
                boundaries.append(boundary())
                write_json(folder / "case.json", dict(**row, measurement=result, settings=settings,
                           segment_wall_ms=segment_wall_ms))
                write_json(ROOT / "boundaries.json", boundaries)
                print("Complete", row["index"], row["arm"], flush=True)
                assert monitor.poll() is None
        finally:
            monitor.terminate()
            monitor.wait(timeout=10)
            metadata.update(end_ns=time.time_ns(), monitor_stopped=monitor.poll() is not None)
            write_json(ROOT / "telemetry_metadata.json", metadata)
    for field in ("sources", "input_hashes", "maintained_files"):
        hashes(p[field])

if __name__ == "__main__":
    run()
"""
(R / "worker.py").write_text(w)
launcher = Path("results/partial_offload_training_v1/launch.py").read_text()
start = launcher.index('if stage == "train":')
end = launcher.index("env = os.environ.copy()")
launcher = launcher[:start] + launcher[end:]
launcher = launcher.replace("partial_offload_training_v1", "interleaved_training_v1")
(R / "launch.py").write_text(launcher)
a = (
    Path("results/partial_offload_training_v1/audit.py")
    .read_text()
    .replace("partial_offload_training_v1", "interleaved_training_v1")
)
a = (
    a.replace('("buffer0", "buffer4", "buffer8")', '("buffer4",)')
    .replace("native_scores=16", "native_scores=24")
    .replace("batches=480", "batches=720")
)
a = a.replace(
    'dict(dataset=f["dataset"], arm=arm, checks=',
    'dict(dataset=f["dataset"], seed=f["seed"], arm=arm, checks=',
)
(R / "audit.py").write_text(a)
