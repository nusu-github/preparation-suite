import collections
import mmap
import os
from collections.abc import Iterable
from datetime import timedelta, timezone
from pathlib import Path

import fiftyone as fo
import fiftyone.types as fot
from fiftyone import ViewField as F
from joblib import Parallel, delayed
from xxhash import xxh3_64


class CreateDataSet:
    time_zone = timezone(timedelta(hours=+9), "JST")

    def __init__(self, data_dir: Path, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.dataset = fo.Dataset.from_dir(
            data_dir,
            dataset_type=fot.MediaDirectory,
        )

        for sample in self.dataset.select_fields("filepath").iter_samples(
            autosave=True,
        ):
            sample["original_filepath"] = sample.filepath

    @staticmethod
    def _get_hashes(files: Iterable, max_workers=-1):
        def _get_hash(file: str):
            with open(file, "rb") as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mapped_file:
                x = xxh3_64()
                x.update(mapped_file)

            return x.hexdigest()

        return Parallel(n_jobs=max_workers, prefer="threads")(delayed(_get_hash)(file) for file in files)

    @staticmethod
    def _color(files: Iterable, max_workers=-1):
        from preparation_suite.module.ImageColor import ImageColor

        image_color = ImageColor()

        return image_color.process_batch(list(files), max_workers=max_workers)

    def delete_duplicates(self):
        hash_list = dict(
            zip(
                self._get_hashes(
                    sample.filepath
                    for sample in self.dataset.select_fields(
                        "filepath",
                    ).iter_samples(progress=True)
                ),
                self.dataset.values("id"),
                strict=True,
            ),
        )

        duplicates = [
            hash_list[file_hash] for file_hash, count in collections.Counter(hash_list.keys()).items() if count > 1
        ]
        self.dataset.delete_samples(duplicates)
        return len(duplicates)

    def export(self, num_workers=None):
        # ThreadPoolExecutor max_workers
        num_workers = num_workers or min(32, (os.cpu_count() or 1) + 4)
        self.dataset.compute_metadata(overwrite=True, num_workers=num_workers)

        output_view = self.dataset.match(F("metadata"))
        output_view.export(
            export_dir=str(self.output_dir),
            dataset_type=fot.FiftyOneDataset,
            use_dirs=True,
        )
