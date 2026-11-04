"""Run a command under CPU and RAM limits without root (cgroups v1 user scopes don't enforce).

- CPU: the process tree is pinned to ``cpus`` cores (kernel-enforced affinity, inherited by
  children) and thread pools are sized to match (``THREAD_ENV``).
- RAM: a watchdog sums the resident memory (RSS) of the whole process tree every ``poll_s``
  seconds and kills the tree (SIGKILL to its process group) above ``mem_gb``. Virtual memory is
  not capped (CUDA reserves large address ranges).
"""

from __future__ import annotations

import os
import signal
import subprocess
import tempfile
import time
from dataclasses import dataclass

import psutil
import structlog

log = structlog.get_logger(__name__)

THREAD_ENV = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_MAX_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "POLARS_MAX_THREADS",
    "RAYON_NUM_THREADS",
)
GB = 1024**3


@dataclass(frozen=True)
class Limits:
    mem_gb: float
    cpus: int | list[int]  # a count (least busy cores are picked) or explicit core ids
    poll_s: float = 0.5


@dataclass(frozen=True)
class LimitedResult:
    returncode: int
    peak_rss_gb: float
    killed_for_memory: bool
    seconds: float
    cpus: list[int]
    stdout: str | None = None
    stderr: str | None = None


def parse_cpus(spec: str) -> list[int]:
    """'0-3,8,10-11' -> [0, 1, 2, 3, 8, 10, 11]."""
    out: list[int] = []
    for part in spec.split(","):
        a, _, b = part.partition("-")
        out.extend(range(int(a), int(b or a) + 1))
    return out


def pick_cpus(n: int, sample_s: float = 0.3) -> list[int]:
    """The ``n`` least busy cores this process may use."""
    allowed = sorted(os.sched_getaffinity(0))
    if n > len(allowed):
        raise ValueError(f"asked for {n} CPUs, only {len(allowed)} available")
    load = psutil.cpu_percent(interval=sample_s, percpu=True)
    return sorted(sorted(allowed, key=lambda c: load[c])[:n])


def _tree_rss(proc: psutil.Process) -> int:
    total = 0
    for p in [proc, *proc.children(recursive=True)]:
        try:
            total += p.memory_info().rss
        except psutil.Error:  # exited in between
            pass
    return total


def run_limited(cmd: list[str], limits: Limits, capture: bool = False) -> LimitedResult:
    cpus = limits.cpus if isinstance(limits.cpus, list) else pick_cpus(limits.cpus)
    env = os.environ | {var: str(len(cpus)) for var in THREAD_ENV}
    out = tempfile.TemporaryFile("w+") if capture else None
    err = tempfile.TemporaryFile("w+") if capture else None
    t0 = time.perf_counter()
    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=out,
        stderr=err,
        start_new_session=True,  # own process group: the watchdog can kill the whole tree
        preexec_fn=lambda: os.sched_setaffinity(0, cpus),
    )
    log.info("limited.start", cmd=" ".join(cmd), cpus=cpus, mem_gb=limits.mem_gb, pid=proc.pid)
    watched, peak, killed = psutil.Process(proc.pid), 0, False
    cap = limits.mem_gb * GB
    while proc.poll() is None:
        try:
            rss = _tree_rss(watched)
        except psutil.NoSuchProcess:
            break
        peak = max(peak, rss)
        if rss > cap:
            killed = True
            log.error("limited.memory_exceeded", rss_gb=round(rss / GB, 2), cap_gb=limits.mem_gb)
            os.killpg(proc.pid, signal.SIGKILL)
            break
        time.sleep(limits.poll_s)
    returncode = proc.wait()
    seconds = time.perf_counter() - t0

    def read(f) -> str | None:
        if f is None:
            return None
        f.seek(0)
        return f.read()

    result = LimitedResult(
        returncode=returncode,
        peak_rss_gb=peak / GB,
        killed_for_memory=killed,
        seconds=seconds,
        cpus=cpus,
        stdout=read(out),
        stderr=read(err),
    )
    log.info(
        "limited.done",
        returncode=returncode,
        peak_rss_gb=round(result.peak_rss_gb, 2),
        killed_for_memory=killed,
        seconds=round(seconds, 1),
    )
    return result
