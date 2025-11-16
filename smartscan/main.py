#! /usr/bin/env python3

import os
import argparse
import asyncio
from smartscan.utils.file import load_dir_list, get_files_from_dirs
from smartscan.constants import  DINO_V2_SMALL_MODEL_PATH, MINILM_MODEL_PATH, SCAN_HISTORY_DB
from smartscan.organiser.analyser import FileAnalyser
from smartscan.organiser.scanner import FileScanner
from smartscan.ml.providers.embeddings.minilm.text import MiniLmTextEmbedder
from smartscan.ml.providers.embeddings.dino.image import DinoSmallV2ImageEmbedder


async def main():
    if not os.path.exists(MINILM_MODEL_PATH):
        raise ValueError(f"Text encoder model not found: {MINILM_MODEL_PATH} ")

    if not os.path.exists(DINO_V2_SMALL_MODEL_PATH):
        raise ValueError(f"Image encoder model not found: {DINO_V2_SMALL_MODEL_PATH}")

    file_analyser = FileAnalyser(
        image_encoder=DinoSmallV2ImageEmbedder(DINO_V2_SMALL_MODEL_PATH),
        text_encoder=MiniLmTextEmbedder(MINILM_MODEL_PATH),
        similarity_threshold=0.4
    )

    file_analyser.image_encoder.init()
    file_analyser.text_encoder.init()


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
        default=0.4,
        help="Similarity threshold for the scan mode. Default is 0.4."
    )

    args = parser.parse_args()

    if args.command == "compare":
        if args.file:
            filepath1, filepath2 = args.file
            if not os.path.isfile(filepath1) or not os.path.isfile(filepath2):
                raise argparse.ArgumentTypeError(f"One or both files not found: {filepath1}, {filepath2}")
            similarity_score = file_analyser.compare_files(filepath1, filepath2)
            print(f"File-to-file similarity: {similarity_score}")

        elif args.dir:
            filepath, dirpath = args.dir
            if not os.path.isfile(filepath) or not os.path.isdir(dirpath):
                raise argparse.ArgumentTypeError(f"Invalid file or directory: {filepath}, {dirpath}")
            similarity_score = file_analyser.compare_file_to_dir(filepath, dirpath)
            print(f"File-to-directory similarity: {similarity_score}")

        elif args.dirs:
            filepath, dirlist_file = args.dirs
            if not os.path.isfile(filepath) or not os.path.isfile(dirlist_file):
                raise argparse.ArgumentTypeError(f"Invalid file or directory list: {filepath}, {dirlist_file}")
            dirpaths = load_dir_list(dirlist_file)
            dirs_similarities = file_analyser.compare_file_to_dirs(filepath, dirpaths)
            print(f"File-to-directories similarity\n--------------------------\n")
            for (key, value) in sorted(dirs_similarities.items(), reverse=True):
                print(f"Directory: {key} | Similarity: {value}")

    elif args.command == "scan":
        file_analyser.similarity_threshold = args.threshold
        if not os.path.isfile(args.target_file) or not os.path.isfile(args.destination_file):
            raise argparse.ArgumentTypeError(f"Target or destination file not found: {args.target_file}, {args.destination_file}")
        
        target_dirs = load_dir_list(args.target_file)
        destination_dirs = load_dir_list(args.destination_file)
        file_scanner = FileScanner(analyser=file_analyser, destination_dirs=destination_dirs, db_path=SCAN_HISTORY_DB)
        await file_scanner.run(get_files_from_dirs(target_dirs))

if __name__ == "__main__":
    asyncio.run(main())
