from __future__ import annotations

import os
import selectors
import signal
import subprocess
import time
from pathlib import Path
from typing import Mapping


def run_bounded_process(
    argv: list[str],
    *,
    cwd: str | Path,
    timeout: float,
    max_output: int,
    env: Mapping[str, str] | None = None,
) -> tuple[int, str, str]:
    if timeout <= 0:
        raise ValueError("Process timeout must be positive.")
    if max_output < 1:
        raise ValueError("Process output limit must be positive.")

    process = subprocess.Popen(
        argv,
        cwd=str(cwd),
        env=dict(env) if env is not None else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=False,
        start_new_session=True,
    )
    if process.stdout is None or process.stderr is None:
        process.kill()
        raise RuntimeError("Could not capture subprocess output.")

    selector = selectors.DefaultSelector()
    streams = {
        process.stdout: bytearray(),
        process.stderr: bytearray(),
    }
    for stream in streams:
        selector.register(stream, selectors.EVENT_READ)
    deadline = time.monotonic() + timeout
    captured = 0

    try:
        while selector.get_map():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"Process timed out after {timeout:g} seconds.")
            events = selector.select(min(remaining, 0.1))
            for key, _ in events:
                allowance = max_output + 1 - captured
                chunk = os.read(key.fd, min(4096, allowance))
                if not chunk:
                    selector.unregister(key.fileobj)
                    continue
                captured += len(chunk)
                if captured > max_output:
                    raise ValueError(
                        f"Process output exceeded the {max_output}-byte limit."
                    )
                streams[key.fileobj].extend(chunk)
        return_code = process.wait(timeout=max(0.01, deadline - time.monotonic()))
    except (TimeoutError, ValueError, subprocess.TimeoutExpired):
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=1)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
        raise
    finally:
        selector.close()
        process.stdout.close()
        process.stderr.close()

    stdout = streams[process.stdout].decode("utf-8", errors="replace")
    stderr = streams[process.stderr].decode("utf-8", errors="replace")
    return return_code, stdout, stderr
