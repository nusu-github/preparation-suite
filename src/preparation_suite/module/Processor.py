from pathlib import Path

from joblib import Parallel, delayed


class Processor:
    def __init__(self) -> None:
        pass

    def process(self, *args, **kwargs):
        raise NotImplementedError

    def process_batch(self, *args, **kwargs):
        raise NotImplementedError

    def process_pipeline(self, *args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def _to_path(item: str | Path) -> Path:
        if isinstance(item, str):
            item = Path(item)

        return item

    @staticmethod
    def _to_paths(items: list[str | Path] | tuple[str | Path]) -> list[Path]:
        return [Processor._to_path(item) for item in items]

    @staticmethod
    def _map(func: callable, items: list, max_workers: int = -1, verbose: bool = False):
        return Parallel(n_jobs=max_workers, return_as="list", prefer="threads", verbose=1 if verbose else 0)(
            delayed(func)(item) for item in items
        )

    @staticmethod
    def _map_process(func: callable, items: list, max_workers: int = -1, verbose: bool = False):
        return Parallel(n_jobs=max_workers, return_as="list", prefer="processes", verbose=1 if verbose else 0)(
            delayed(func)(item) for item in items
        )

    @staticmethod
    def _iter(func: callable, items: list, max_workers: int = -1, verbose: bool = False):
        return Parallel(n_jobs=max_workers, return_as="generator", prefer="threads", verbose=1 if verbose else 0)(
            delayed(func)(item) for item in items
        )

    @staticmethod
    def _iter_process(func: callable, items: list, max_workers: int = -1, verbose: bool = False):
        return Parallel(n_jobs=max_workers, return_as="generator", prefer="processes", verbose=1 if verbose else 0)(
            delayed(func)(item) for item in items
        )
