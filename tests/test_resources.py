import os
import sys

from battery_optimisation.resources import Limits, parse_cpus, pick_cpus, run_limited

PY = sys.executable


def test_parse_cpus() -> None:
    assert parse_cpus("0-3,8,10-11") == [0, 1, 2, 3, 8, 10, 11]


def test_pick_cpus_takes_the_requested_count() -> None:
    cpus = pick_cpus(4)
    assert len(cpus) == 4 and set(cpus) <= os.sched_getaffinity(0)


def test_cpu_affinity_and_thread_env_are_applied() -> None:
    code = (
        "import os; print(len(os.sched_getaffinity(0)), os.environ['OMP_NUM_THREADS'], "
        "os.environ['POLARS_MAX_THREADS'])"
    )
    result = run_limited([PY, "-c", code], Limits(mem_gb=2, cpus=2), capture=True)
    assert result.returncode == 0
    assert result.stdout.split() == ["2", "2", "2"]


def test_memory_cap_kills_the_process_tree() -> None:
    # child allocates and touches 600 MB under a 0.2 GB cap -> killed by the watchdog
    code = (
        "x = bytearray(600 * 1024 * 1024); x[::4096] = b'1' * len(x[::4096]); "
        "import time; time.sleep(5)"
    )
    result = run_limited([PY, "-c", code], Limits(mem_gb=0.2, cpus=1), capture=True)
    assert result.killed_for_memory
    assert result.returncode != 0
    assert result.peak_rss_gb >= 0.2


def test_within_limits_runs_normally() -> None:
    result = run_limited([PY, "-c", "print('ok')"], Limits(mem_gb=1, cpus=1), capture=True)
    assert (result.returncode, result.stdout.strip(), result.killed_for_memory) == (0, "ok", False)
