from pathlib import Path

from preparation_suite import DATA_DIR, DATASET_DIR
from preparation_suite.CreateDataSet import CreateDataSet


def main(data_dir: Path, output_dir: Path):
    # load data
    print("Loading data...")
    dataset = CreateDataSet(data_dir, output_dir)

    # delete duplicates
    print("Deleting duplicates...")
    duplicates_count = dataset.delete_duplicates()
    print(f"Deleting {duplicates_count} duplicates...")

    # export
    print("Exporting...")
    dataset.export()


if __name__ == "__main__":
    import argparse

    default_data_dir = DATA_DIR / "data"
    default_output_dir = DATASET_DIR / default_data_dir.name

    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, default=default_data_dir, help="data directory")
    parser.add_argument("--output_dir", type=str, default=default_output_dir, help="output directory")

    args = parser.parse_args()

    main(Path(args.data_dir), Path(args.output_dir))
