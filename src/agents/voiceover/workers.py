"""Fan-out helpers for rendering narration lines in parallel processes.

Pure scheduling: how many workers, how to cut the lines, how to seed a worker's
RNG, and how to collect results. The per-line synthesis callback is passed in,
so this module knows nothing about the TTS engines themselves.
"""

import os
import multiprocessing as mp

# Each worker holds its own ~4.3GB model copy: 2 gives ~1.9x wall-clock, 3+ risks
# RAM on a 14GB box.
TTS_WORKERS = int(os.getenv("TTS_WORKERS", "2"))
# Chatterbox 0.1.7 has no rate knob, so this is an ffmpeg atempo stretch applied
# to each line (pitch preserved). 1.0 is the model's natural pace; 0.93 suits a
# reflective channel better than a rushed one.


def _effective_workers(n_lines: int) -> int:
    """Number of parallel render workers (settings.TTS_WORKERS, default 2).

    Measured on this machine: each worker holds ~4.3GB (own model copy) and a
    single worker already uses all memory bandwidth, so 2 workers give ~1.9x
    wall-clock speedup at identical quality; 3+ would exceed RAM on 14GB boxes.
    """
    if not n_lines or n_lines < 2:
        return 1
    return min(TTS_WORKERS, 2, n_lines)


def _split_chunks(lines: list, workers: int) -> list:
    """Balanced contiguous chunks preserving original line order."""
    n = len(lines)
    if n <= workers:
        return [[line] for line in lines]
    base, rem = divmod(n, workers)
    chunks, i = [], 0
    for w in range(workers):
        size = base + (1 if w < rem else 0)
        chunks.append(lines[i:i + size])
        i += size
    return chunks


def _setup_worker_rng(seed: int, threads: int) -> None:
    """Cap each worker's torch threads and re-seed RNG so concurrent lines
    get independent samples without oversubscribing the CPU."""
    import numpy as _np
    import torch as _torch
    _torch.set_num_threads(threads)
    _torch.manual_seed(seed)
    _np.random.seed(seed)

def _line_voice_key(line: dict, default_voice: dict) -> str:
    """Identity of the character speaking a line, for grouping."""
    voice = line.get("voice") or default_voice or {}
    return voice.get("ref_audio") or voice.get("name") or "default"


def _split_chunks_by_voice(lines: list, workers: int,
                           default_voice: dict = None) -> list:
    """Split lines across workers, keeping each character's lines together.

    A 2-character video has 2 lines and 2 workers, so this hands one character
    to each worker: both synthesise a line in parallel and neither has to load
    the other's voice conditionals. Splitting by line count instead would let
    both workers land on the same character and serialise on one conditionals
    load.

    When there are fewer distinct characters than workers (a 3-line monologue),
    there is nothing to keep apart, so this falls back to a plain even split
    and the workers duplicate the conditionals load, as they always have.
    """
    groups: dict = {}
    for line in lines:
        groups.setdefault(_line_voice_key(line, default_voice), []).append(line)

    if len(groups) < workers:
        return _split_chunks(lines, workers)

    bins: list = [[] for _ in range(workers)]
    loads = [0] * workers
    # Largest group first, so the big character's lines land in the emptiest bin.
    for key in sorted(groups, key=lambda k: -len(groups[k])):
        target = loads.index(min(loads))
        bins[target].extend(groups[key])
        loads[target] += len(groups[key])
    return [b for b in bins if b]


def _parallel_lines(workers, worker_fn, lines, audio_dir, default_voice=None) -> list:
    """Render `workers` disjoint chunks of lines in parallel subprocesses.

    Uses spawn: each worker is a fresh interpreter holding its own model. Torch is
    not fork-safe once its thread pools exist (forking the parent's loaded model
    caused hangs), and two copies still fit comfortably in RAM."""
    ctx = mp.get_context("spawn")
    queue = ctx.SimpleQueue()
    procs = []
    threads = max(1, (os.cpu_count() or 4) // workers)
    for idx, chunk in enumerate(_split_chunks_by_voice(lines, workers, default_voice)):
        seed = ((os.getpid() << 16) ^ ((idx + 1) * 7919)) & 0xFFFFFFFF
        p = ctx.Process(target=worker_fn,
                        args=(idx, chunk, audio_dir, default_voice, seed, threads, queue))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()
    errors, results = [], {}
    for _ in procs:
        err, idx, payload = queue.get()
        if err:
            errors.append(payload)
        else:
            results[idx] = payload
    if errors:
        raise RuntimeError("; ".join(str(e) for e in errors))
    out_lines = []
    for idx in sorted(results):
        out_lines.extend(results[idx])
    # Workers return deep copies (pickled across processes); fold the new
    # audio_path/actual_duration back into the original dict objects so callers
    # that rely on in-place mutation keep working. Matched by line id, not
    # position: grouping by character means a chunk's lines are not contiguous,
    # so a positional zip would write one line's audio onto another's.
    by_id = {ln["id"]: ln for ln in lines}
    for upd in out_lines:
        orig = by_id.get(upd["id"])
        if orig is not None:
            orig.update(upd)
    return lines
