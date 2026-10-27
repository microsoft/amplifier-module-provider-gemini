"""Exercise the real conftest and pytest-asyncio in a fresh interpreter."""

import json
import os
import subprocess
import sys
from pathlib import Path

_CHILD = """\
import asyncio
import json
import weakref

import pytest

records, references, roles = [], [], []
generations, observed_classes, finalizing = {}, set(), set()
original_new = asyncio.DefaultEventLoopPolicy.new_event_loop
original_del = asyncio.BaseEventLoop.__del__


def observe_close(loop_type):
    original_close = loop_type.close

    def close(loop):
        generation = generations.get(id(loop))
        original_close(loop)
        if generation is not None:
            records[generation]["closes"].append({
                "finalizer": generation in finalizing,
                "closed": loop.is_closed(),
            })

    loop_type.close = close


def new_loop(policy):
    loop = original_new(policy)
    generation = len(records)
    generations[id(loop)] = generation
    references.append(weakref.ref(loop))
    records.append({"generation": generation, "id": id(loop), "closes": []})
    if type(loop) not in observed_classes:
        observe_close(type(loop))
        observed_classes.add(type(loop))
    return loop


def finalize(loop):
    generation = generations.get(id(loop))
    finalizing.add(generation)
    try:
        original_del(loop)
    finally:
        finalizing.discard(generation)


asyncio.DefaultEventLoopPolicy.new_event_loop = new_loop
asyncio.BaseEventLoop.__del__ = finalize
# Deliberately do not initialize this caller-owned policy's loop slot.
original = asyncio.DefaultEventLoopPolicy()
asyncio.set_event_loop_policy(original)
exitcode = pytest.main([
    "-q", "-c", "pytest.ini", "--confcutdir=.",
    "-p", "pytest_asyncio.plugin", "test_lifecycle.py",
])
# Freeze close evidence immediately; later finalizers cannot repair it.
snapshot = [
    dict(record, closes=list(record["closes"]), alive=references[i]() is not None)
    for i, record in enumerate(records)
]
restored = asyncio.get_event_loop_policy() is original
before_probe = len(records)
probe_raised, probe_closed = False, False
try:
    probe = original.get_event_loop()
except RuntimeError:
    probe_raised = True
else:
    # Close only a loop newly allocated by this post-snapshot probe.
    if len(records) == before_probe + 1:
        probe.close()
        probe_closed = probe.is_closed()
    del probe
print("LOOP_POLICY_RESULT=" + json.dumps({
    "pytest_exitcode": int(exitcode), "loops": snapshot, "roles": roles,
    "restored": restored, "probe_new_loops": len(records) - before_probe,
    "probe_raised": probe_raised, "probe_closed": probe_closed,
}))
"""

_INNER_TESTS = """\
import asyncio
from __main__ import generations, roles

import pytest


async def record_loop(role):
    roles.append([role, generations[id(asyncio.get_running_loop())]])


@pytest.mark.asyncio
async def test_01_runner():
    await record_loop("runner")


def test_02_asyncio_run():
    asyncio.run(record_loop("asyncio.run"))
"""


def test_event_loop_policy_does_not_create_an_unowned_loop(tmp_path):
    """Runner teardown must not leave an implicit loop for run() to displace."""
    (tmp_path / "conftest.py").write_bytes(
        Path(__file__).with_name("conftest.py").read_bytes()
    )
    (tmp_path / "pytest.ini").write_text(
        "[pytest]\nasyncio_mode = strict\n", encoding="utf-8"
    )
    (tmp_path / "test_lifecycle.py").write_text(_INNER_TESTS, encoding="utf-8")
    env = {**os.environ, "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"}
    env.pop("PYTEST_ADDOPTS", None)
    env.pop("PYTEST_PLUGINS", None)
    child = subprocess.run(
        [sys.executable, "-c", _CHILD],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    output = child.stdout + child.stderr
    assert child.returncode == 0, output
    prefix = "LOOP_POLICY_RESULT="
    reports = [
        line.removeprefix(prefix)
        for line in child.stdout.splitlines()
        if line.startswith(prefix)
    ]
    assert len(reports) == 1, output
    report = json.loads(reports[0])
    assert report["pytest_exitcode"] == 0, output
    # Baseline creates three loops, regardless of when its orphan is collected.
    assert len(report["loops"]) == 2, report
    assert report["roles"] == [["runner", 0], ["asyncio.run", 1]], report
    assert all(
        any(close == {"finalizer": False, "closed": True} for close in loop["closes"])
        for loop in report["loops"]
    ), report
    assert report["restored"], report
    if sys.version_info < (3, 14):
        assert report["probe_new_loops"] == 1, report
        assert report["probe_closed"], report
        assert not report["probe_raised"], report
    else:
        # Python 3.14 removed implicit allocation; exercise compatibility, not skip.
        assert report["probe_new_loops"] == 0, report
        assert report["probe_raised"], report
    assert "ResourceWarning" not in output, output
