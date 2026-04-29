from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed, Future
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class ParallelResult:
    succeeded: list[Any]               = field(default_factory=list)
    failed:    list[dict[str, str]]    = field(default_factory=list)

    @property
    def all_succeeded(self) -> bool:
        return len(self.failed) == 0

    @property
    def failure_rate(self) -> float:
        total = len(self.succeeded) + len(self.failed)
        return len(self.failed) / total if total else 0.0


def parallel(
    *callables: Callable,
    max_workers: int   = 8,
    timeout:     float = 30.0,
) -> ParallelResult:
    """
    Run callables concurrently in a thread pool, collect typed results
    and partial failures cleanly.

    Usage:
        result = parallel(
            lambda: check_spam(email),
            lambda: check_language(email),
        )
        if not result.all_succeeded:
            handle(result.failed)
        spam, language = result.succeeded
    """
    result = ParallelResult()

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        future_to_name: dict[Future, str] = {
            pool.submit(fn): getattr(fn, "__name__", f"callable_{i}")
            for i, fn in enumerate(callables)
        }

        for future in as_completed(future_to_name, timeout=timeout):
            name = future_to_name[future]
            exc  = future.exception()
            if exc is None:
                result.succeeded.append(future.result())
            else:
                result.failed.append({"key": name, "error": str(exc)})

    return result


def parallel_map(
    fn:          Callable,
    items:       list[Any],
    max_workers: int   = 8,
    timeout:     float = 60.0,
) -> ParallelResult:
    """
    Apply fn to each item in parallel.

    Usage:
        result = parallel_map(fabric_data_agent, dax_plan.queries)
    """
    callables = [functools.partial(fn, item) for item in items]
    return parallel(*callables, max_workers=max_workers, timeout=timeout)


import functools  # noqa: E402 — imported after use in parallel_map body
