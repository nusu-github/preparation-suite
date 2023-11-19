import argparse
from pathlib import Path

import fiftyone as fo
from fiftyone import ViewField as F
from fiftyone.types import FiftyOneDataset
from preparation_suite import DATASET_DIR, MODEL_DIR
from preparation_suite.Preprocessor import Preprocessor


def main(
    data_dir,
    output_dir,
    batch_size,
    aesthetic,
    del_duplicate,
    set_uniqueness,
    img_hash,
    color,
    object_dct,
    dino,
    wd_tag,
):
    dataset = fo.Dataset.from_dir(data_dir, dataset_type=FiftyOneDataset)

    preproc = Preprocessor(dataset, batch_size)

    if img_hash:
        # compute image hash
        print("Computing image hash...")
        preproc.img_hash()

    if set_uniqueness:
        # set uniqueness
        print("Setting uniqueness...")
        preproc.uniqueness()

    if del_duplicate:
        # delete duplicate
        print("Deleting duplicates...")
        preproc.del_duplicate()

    if color:
        # color
        print("Setting color...")
        preproc.color()

    if aesthetic:
        # clip aesthetic
        print("Clip aesthetic...")
        preproc.clip_aesthetic(MODEL_DIR / "sac_public_2022_06_29_vit_l_14_linear.pth")

    if object_dct:
        # object detection
        print("Object detection...")
        if dino:
            print("Dino object detection...")
            preproc.dino_object_detection()

    if wd_tag:
        # wd tag
        print("Setting wd tag...")
        preproc.wd_tags_label()

    output_view = dataset.match(F("metadata"))
    output_view.export(export_dir=str(output_dir), dataset_type=FiftyOneDataset, export_media=False)


def get_parser():
    # Create a parser for command line arguments.
    parser = argparse.ArgumentParser(description="Data preprocessing script with various options")

    # Set default paths.
    default_data_dir = DATASET_DIR / "data"
    default_output_dir = DATASET_DIR / f"preprocessed_{default_data_dir.name}"

    # Add arguments related to directory settings.
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=default_data_dir,
        help="Directory containing the data to process",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=default_output_dir,
        help="Directory where the output should be stored",
    )

    # Add arguments related to processing parameters.
    parser.add_argument("--batch_size", type=int, default=16, help="Number of data items to process at a time")
    parser.add_argument("--aesthetic", action="store_true", default=True, help="Enable aesthetic processing")
    parser.add_argument(
        "--del_duplicate",
        action="store_true",
        default=False,
        help="Remove duplicates during processing",
    )
    parser.add_argument("--set_image_hash", action="store_true", default=True, help="Apply image hashing")
    parser.add_argument("--set_color", action="store_true", default=True, help="Set specific color processing")
    parser.add_argument(
        "--set_uniqueness",
        action="store_true",
        default=False,
        help="Enforce uniqueness in data processing",
    )
    parser.add_argument("--object_dct", action="store_true", default=True, help="Enable object detection")
    parser.add_argument("--dino", action="store_true", default=True, help="Enable dino object detection")
    parser.add_argument("--wd_tag", action="store_true", default=True, help="Include watermarked tagging")

    return parser


if __name__ == "__main__":
    # Parse the arguments.
    parser = get_parser()
    args = parser.parse_args()

    # Call the main function with parsed arguments.
    main(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        aesthetic=args.aesthetic,
        del_duplicate=args.del_duplicate,
        set_uniqueness=args.set_uniqueness,
        img_hash=args.set_image_hash,
        color=args.set_color,
        object_dct=args.object_dct,
        dino=args.dino,
        wd_tag=args.wd_tag,
    )
