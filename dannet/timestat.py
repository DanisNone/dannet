import sys
import time
import threading

ENABLED = False

_lock = threading.Lock()
_stats = {}


class Timer:
    def __init__(self, name: str):
        self.name = name
        self.start_time = None

    def __enter__(self):
        if ENABLED:
            self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not ENABLED or self.start_time is None:
            return
        end_time = time.perf_counter()
        duration = end_time - self.start_time
        with _lock:
            if self.name not in _stats:
                _stats[self.name] = []
            _stats[self.name].append(duration)


def record(name: str) -> Timer:
    return Timer(name)


def get_stats() -> dict:
    with _lock:
        return dict(_stats)


def print_stats(out=None):
    if out is None:
        out = sys.stdout

    with _lock:
        print('Time statistics:', file=out)
        stats = sorted(
            _stats.items(),
            key=lambda nt: sum(nt[1])/len(nt[1])
        )
        for name, times in stats:
            total = sum(times) / len(times)
            print(
                f'{total * 1000:8.3f} ms; '
                f'{len(times)} calls; '
                f'{name:20}',
                file=out
            )


def reset():
    with _lock:
        _stats.clear()


def enable():
    global ENABLED
    ENABLED = True


def disable():
    global ENABLED
    ENABLED = False


def enabled():
    return ENABLED
