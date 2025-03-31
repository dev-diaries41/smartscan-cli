#! /usr/bin/env python3

import os
import open_clip
import argparse
from compare import compare_files, compare_file_to_dir, compare_file_to_dirs, scan
from utils import load_dir_list, move_file
from onnx_utils.convert import text_encoder_to_onnx, image_encoder_to_onnx
from constants import MODEL_NAME, PRETRAINED, TEXT_ENCODER_PATH, IMAGE_ENCODER_PATH



def on_threshold_reached(file_path: str, target_dir: str):
    move_file(file_path, target_dir)

def main():
    if not os.path.exists(TEXT_ENCODER_PATH):
        print("Text encoder model not found, creating it now...")
        text_encoder_to_onnx(MODEL_NAME, PRETRAINED, TEXT_ENCODER_PATH)

    if not os.path.exists(IMAGE_ENCODER_PATH):
        print("Image encoder model not found, creating it now...")
        image_encoder_to_onnx(MODEL_NAME, PRETRAINED, IMAGE_ENCODER_PATH)
        
            
    parser = argparse.ArgumentParser(
        description="CLI tool for comparing text embeddings and scanning directories."
    )
    subparsers = parser.add_subparsers(dest='command', required=True)

    # Compare subcommand
    compare_parser = subparsers.add_parser("compare", help="Perform comparisons using different modes.")
    compare_group = compare_parser.add_mutually_exclusive_group(required=True)
    compare_group.add_argument(
        "-f", "--file",
        nargs=2,
        metavar=("FILEPATH1", "FILEPATH2"),
        help="Compare two text files."
    )
    compare_group.add_argument(
        "-d", "--dir",
        nargs=2,
        metavar=("FILEPATH", "DIRPATH"),
        help="Compare a text file against a directory."
    )
    compare_group.add_argument(
        "-l", "--dirs",
        nargs=2,
        metavar=("FILEPATH", "DIRLISTFILE"),
        help="Compare a text file against multiple directories listed in a file."
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
    tokenizer = open_clip.get_tokenizer('ViT-B-32')
    # Uncomment this line if you do not wish to use the quantized onnx model then add model to all the 4 function below using model=model
    # model, _, _ = open_clip.create_model_and_transforms(MODEL_NAME, pretrained=PRETRAINED)

    if args.command == "compare":
        if args.file:
            filepath1, filepath2 = args.file
            if not os.path.isfile(filepath1) or not os.path.isfile(filepath2):
                raise argparse.ArgumentTypeError(f"One or both files not found: {filepath1}, {filepath2}")
            similarity_score = compare_files(filepath1, filepath2, tokenizer)
            print(f"File-to-file similarity: {similarity_score}")

        elif args.dir:
            filepath, dirpath = args.dir
            if not os.path.isfile(filepath) or not os.path.isdir(dirpath):
                raise argparse.ArgumentTypeError(f"Invalid file or directory: {filepath}, {dirpath}")
            similarity_score = compare_file_to_dir(filepath, dirpath, tokenizer)
            print(f"File-to-directory similarity: {similarity_score}")

        elif args.dirs:
            filepath, dirlist_file = args.dirs
            if not os.path.isfile(filepath) or not os.path.isfile(dirlist_file):
                raise argparse.ArgumentTypeError(f"Invalid file or directory list: {filepath}, {dirlist_file}")
            dirpaths = load_dir_list(dirlist_file)
            best_dirpath, best_similarity = compare_file_to_dirs(filepath, dirpaths, tokenizer)
            print(f"Best matching directory: {best_dirpath} with similarity: {best_similarity}")

    elif args.command == "scan":
        if not os.path.isfile(args.target_file) or not os.path.isfile(args.destination_file):
            raise argparse.ArgumentTypeError(f"Target or destination file not found: {args.target_file}, {args.destination_file}")
        
        target_dirs = load_dir_list(args.target_file)
        destination_dirs = load_dir_list(args.destination_file)
        scan(target_dirs, destination_dirs, tokenizer, on_threshold_reached, threshold=args.threshold)

if __name__ == "__main__":
    main()
