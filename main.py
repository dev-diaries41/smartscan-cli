#! /usr/bin/env python3

import os
import argparse
from utils import load_dir_list, move_file
from constants import TEXT_ENCODER_PATH, IMAGE_ENCODER_PATH, MINILM_MODEL_PATH
from organiser import FileOrganiser
from ml.models.providers.embeddings.clip.image import ClipImageEmbedder
from ml.models.providers.embeddings.minilm.text import MiniLmTextEmbedder


def on_threshold_reached(file_path: str, target_dir: str):
    move_file(file_path, target_dir)

def main():
    if not os.path.exists(MINILM_MODEL_PATH):
        raise ValueError(f"Text encoder model not found: {MINILM_MODEL_PATH} ")

    if not os.path.exists(IMAGE_ENCODER_PATH):
        raise ValueError(f"Image encoder model not found: {IMAGE_ENCODER_PATH}")

    file_organiser = FileOrganiser(
        image_encoder=ClipImageEmbedder(IMAGE_ENCODER_PATH),
        text_encoder=MiniLmTextEmbedder(MINILM_MODEL_PATH),
        similarity_threshold=0.52
    )

    file_organiser.image_encoder.init()
    file_organiser.text_encoder.init()

    parser = argparse.ArgumentParser(
        description="CLI tool for comparing text or image embeddings and scanning directories."
    )
    subparsers = parser.add_subparsers(dest='command', required=True)

    # Compare subcommand
    compare_parser = subparsers.add_parser("compare", help="Perform comparisons using different modes.")
    compare_group = compare_parser.add_mutually_exclusive_group(required=True)
    compare_group.add_argument(
        "-f", "--file",
        nargs=2,
        metavar=("FILEPATH1", "FILEPATH2"),
        help="Compare two text or image files."
    )
    compare_group.add_argument(
        "-d", "--dir",
        nargs=2,
        metavar=("FILEPATH", "DIRPATH"),
        help="Compare a text or image file against a directory."
    )
    compare_group.add_argument(
        "-l", "--dirs",
        nargs=2,
        metavar=("FILEPATH", "DIRLISTFILE"),
        help="Compare a text or image file against multiple directories listed in a file."
    )

    # Scan subcommand
    scan_parser = subparsers.add_parser("scan", help="Scan directories and auto organise files into directories.")
    scan_parser.add_argument(
        "target_file", metavar="TARGET_FILE", help="File listing target directories."
    )
    scan_parser.add_argument(
        "destination_file", metavar="DESTINATION_FILE", help="File listing destination directories."
    )
    scan_parser.add_argument(
        "-t", "--threshold",
        type=float,
        default=0.7,
        help="Similarity threshold for the scan mode. Default is 0.7."
    )

    args = parser.parse_args()

    if args.command == "compare":
        if args.file:
            filepath1, filepath2 = args.file
            if not os.path.isfile(filepath1) or not os.path.isfile(filepath2):
                raise argparse.ArgumentTypeError(f"One or both files not found: {filepath1}, {filepath2}")
            similarity_score = file_organiser.compare_files(filepath1, filepath2)
            print(f"File-to-file similarity: {similarity_score}")

        elif args.dir:
            filepath, dirpath = args.dir
            if not os.path.isfile(filepath) or not os.path.isdir(dirpath):
                raise argparse.ArgumentTypeError(f"Invalid file or directory: {filepath}, {dirpath}")
            similarity_score = file_organiser.compare_file_to_dir(filepath, dirpath)
            print(f"File-to-directory similarity: {similarity_score}")

        elif args.dirs:
            filepath, dirlist_file = args.dirs
            if not os.path.isfile(filepath) or not os.path.isfile(dirlist_file):
                raise argparse.ArgumentTypeError(f"Invalid file or directory list: {filepath}, {dirlist_file}")
            dirpaths = load_dir_list(dirlist_file)
            best_dirpath, best_similarity = file_organiser.compare_file_to_dirs(filepath, dirpaths)
            print(f"Best matching directory: {best_dirpath} with similarity: {best_similarity}")

    elif args.command == "scan":
        file_organiser.similarity_threshold = args.threshold
        if not os.path.isfile(args.target_file) or not os.path.isfile(args.destination_file):
            raise argparse.ArgumentTypeError(f"Target or destination file not found: {args.target_file}, {args.destination_file}")
        
        target_dirs = load_dir_list(args.target_file)
        destination_dirs = load_dir_list(args.destination_file)
        file_organiser.scan(target_dirs, destination_dirs, on_threshold_reached)

if __name__ == "__main__":
    main()
