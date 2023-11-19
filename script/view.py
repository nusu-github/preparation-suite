import shutil
from pathlib import Path

import fiftyone as fo
from fiftyone import ViewField as F
from preparation_suite import DATASET_DIR


def main(dataset_dir: Path):
    dataset = fo.Dataset.from_dir(
        dataset_dir,
        dataset_type=fo.types.FiftyOneDataset,
    )

    export = False
    if export:
        view = (
            dataset.match(F("character_tags.solo") >= 4.0).match(F("rating") != "explicit").match(F("is_color"))
            # .match(F("uniqueness") >= 0.3)
        ).iter_samples(progress=True)

        output_dir = DATASET_DIR / "qc_ok"

        output_dir.mkdir(parents=True, exist_ok=True)
        for sample in view:
            shutil.copy(sample.filepath, output_dir)
    else:
        session = fo.launch_app(dataset)

        session.wait()


if __name__ == "__main__":
    import argparse

    default_dataset_dir = DATASET_DIR / "preprocessed_data"

    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_dir", type=str, default=default_dataset_dir, help="dataset directory")
    args = parser.parse_args()

    main(Path(args.dataset_dir))
