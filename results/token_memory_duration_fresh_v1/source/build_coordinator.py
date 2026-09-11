"""Prepare a bounded continuation from the preserved first recovery coordinator."""

from pathlib import Path

root = Path("results/token_memory_duration_fresh_v1/source")
old = Path("results/token_memory_duration_recovery_v1/source/study.py").read_text()
old = old.replace("token_memory_duration_recovery_v1", "token_memory_duration_fresh_v1")
start = old.index("    # Two explicit CPU dependency checks")
end = old.index("    rows, pairs = [], []", start)
qualification = """    with (ROOT / "qualification.log").open("x") as log:
        code = subprocess.run(
            command() + ["qualify"], env=os.environ.copy(),
            stdout=log, stderr=subprocess.STDOUT,
        ).returncode
    (ROOT / "qualification_exit.txt").write_text(str(code))
    assert code == 0, "Qualification failed; no training allocated"
"""
old = old[:start] + qualification + old[end:]
start = old.index("                            [\n                                sys.executable")
end = old.index("                            env=os.environ.copy()", start)
old = (
    old[:start]
    + """                            command() + [str(batch), str(context), str(seed), policy],
"""
    + old[end:]
)
point = old.index("\ndef run():")
old = (
    old[:point]
    + """
def command():
    return [
        sys.executable, "-B", "-X",
        "pycache_prefix=" + str(ROOT.resolve() / "unused_worker_cache"),
        "-X", "faulthandler", "-u", "-m",
        "results.token_memory_duration_fresh_v1.source.worker",
    ]

"""
    + old[point:]
)
with (root / "study.py").open("x") as stream:
    stream.write(old)
launcher = Path("results/token_memory_duration_v1/source/launch.py").read_text()
launcher = launcher.replace("token_memory_duration_v1", "token_memory_duration_fresh_v1")
with (root / "launch.py").open("x") as stream:
    stream.write(launcher)
print("Prepared continuation coordinator; scientific worker remains an unchanged import")
