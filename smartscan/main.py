#! /usr/bin/env python3

import os
import argparse
import asyncio
import chromadb
import shutil

from smartscan.utils.file import load_dir_list, get_files_from_dirs
from smartscan.constants import  DINO_V2_SMALL_MODEL_PATH, MINILM_MODEL_PATH, SCAN_HISTORY_DB, DB_DIR
from smartscan.organiser.analyser import FileAnalyser
from smartscan.organiser.scanner import FileScanner
from smartscan.organiser.scanner_listener import FileScannerListener
from smartscan.ml.providers.embeddings.minilm.text import MiniLmTextEmbedder
from smartscan.ml.providers.embeddings.dino.image import DinoSmallV2ImageEmbedder
from smartscan.indexer.indexer import FileIndexer
from smartscan.indexer.indexer_listener import FileIndexerListener
from smartscan.data.scan_history import ScanHistoryDB, ScanHistoryFilterOpts

async def main():
    if not os.path.exists(MINILM_MODEL_PATH):
        raise ValueError(f"Text encoder model not found: {MINILM_MODEL_PATH} ")

    if not os.path.exists(DINO_V2_SMALL_MODEL_PATH):
        raise ValueError(f"Image encoder model not found: {DINO_V2_SMALL_MODEL_PATH}")


    client = chromadb.PersistentClient(path=DB_DIR)
    text_store = client.get_or_create_collection(
        name="text_collection",
        metadata={"description": "Collection for text documents"}
    )
    image_store = client.get_or_create_collection(
        name="image_collection",
        metadata={"description": "Collection for images"}
    ) 
    video_store = client.get_or_create_collection(
        name="test_video_collection",
        metadata={"description": "Collection for videos"}
    )

    file_analyser = FileAnalyser(
        image_encoder=DinoSmallV2ImageEmbedder(DINO_V2_SMALL_MODEL_PATH),
        text_encoder=MiniLmTextEmbedder(MINILM_MODEL_PATH),
        similarity_threshold=0.4,
        image_store=image_store,
        text_store=text_store,
        video_store=video_store,
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

    index_parser = subparsers.add_parser("index", help="Index files from selected directories.")
    index_parser.add_argument(
        "--n_frames",
        type=int,
        default=10,
        help="The number of frames to use when creating embedding for video. Default is 10"
    )
    index_group = index_parser.add_mutually_exclusive_group(required=True)
    index_group.add_argument(
        "dirlist_file",
        nargs='?',
        metavar="DIRLISTFILE",
        help="File listing target directories."
    )
    index_group.add_argument(
        "--dirs",
        nargs='+',
        metavar="DIRLIST",
        help="List of directories to index."
    )


    restore_parser = subparsers.add_parser("restore", help="Restore files that were moved to their original location.")
    restore_parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="The start_date period of when to restore"
    )
    restore_parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="The start_date period of when to restore"
    )
    restore_group = restore_parser.add_mutually_exclusive_group(required=True)
    restore_group.add_argument(
        "file",
        nargs='?',
        metavar="FILETORESTORE",
        help="File to restore"
    )
    restore_group.add_argument(
        "--files",
        nargs='+',
        metavar="FILESTORESTORE",
        help="List of files to restore."
    )

    args = parser.parse_args()

    if args.command == "compare":
        if args.file:
            filepath1, filepath2 = args.file
            if not os.path.isfile(filepath1) or not os.path.isfile(filepath2):
                raise argparse.ArgumentTypeError(f"One or both files not found: {filepath1}, {filepath2}")
            similarity_score = file_analyser.compare_files([filepath1, filepath2])
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
            for (key, value) in sorted(dirs_similarities.items()):
                print(f"Directory: {key} | Similarity: {value}")

    elif args.command == "scan":
        file_analyser.similarity_threshold = args.threshold
        if not os.path.isfile(args.target_file) or not os.path.isfile(args.destination_file):
            raise argparse.ArgumentTypeError(f"Target or destination file not found: {args.target_file}, {args.destination_file}")
        
        target_dirs = load_dir_list(args.target_file)
        destination_dirs = load_dir_list(args.destination_file)
        file_scanner = FileScanner(analyser=file_analyser, destination_dirs=destination_dirs, db_path=SCAN_HISTORY_DB , listener = FileScannerListener())
        allowed_exts = file_analyser.valid_img_exts + file_analyser.valid_txt_exts + file_analyser.valid_vid_exts
        result = await file_scanner.run(get_files_from_dirs(target_dirs, allowed_exts=allowed_exts))
        print(f"Scan results - files moved: {result.total_processed} | time elpased: {result.time_elapsed:.2f}s")
    
    elif args.command == "index":
        indexer = FileIndexer(
            image_encoder=DinoSmallV2ImageEmbedder(DINO_V2_SMALL_MODEL_PATH),
            text_encoder=MiniLmTextEmbedder(MINILM_MODEL_PATH),
            image_store=image_store,
            text_store=text_store,
            video_store=video_store,
            listener = FileIndexerListener()
        )

        if args.dirlist_file:
            if not os.path.isfile(args.dirlist_file):
                raise argparse.ArgumentTypeError(f"Invalid file: {dirlist_file}")
            
            dirpaths = load_dir_list(args.dirlist_file)
        elif args.dirs:
            dirpaths = args.dirs
        allowed_exts = indexer.valid_img_exts + indexer.valid_txt_exts + indexer.valid_vid_exts
        files = get_files_from_dirs(dirpaths, allowed_exts=allowed_exts)

        indexer.image_encoder.init()
        indexer.text_encoder.init()
        _ = await indexer.run(indexer.filter(files))

    elif args.command == "restore":
        db = ScanHistoryDB(SCAN_HISTORY_DB)

        if args.file:
            original_source = db.get_original_source(args.file)
            if not original_source:
                print("Original source file not found")
                return
            shutil.move(args.file, original_source)
            print(f"File restored successfully: {original_source}")
        elif args.files:
            source_files = [db.get_original_source(df) for df in args.files]
            restore_failed = []
            restored_count = 0
            for idx, source in enumerate(source_files):
                if source:
                    try:
                        shutil.move(args.files[idx], source)
                        restored_count += 1
                    except Exception:
                        restore_failed.append(source) 
                else:
                    restore_failed.append(source) 
            print(f"{restored_count} files restored successfully")
            for invalid_file in restore_failed:
                print(f"Failed to restore: {invalid_file}")
        elif args.start_date or args.end_date:
            scans = db.get(filter_opts=ScanHistoryFilterOpts(start_date=args.start_date, end_date=args.end_date))
            if not scans:
                print(f"No scan history found matching dates")
                return
            destination_files = set([scan['destination_file'] for scan in scans])
            source_files = [db.get_original_source(df) for df in destination_files]
            restore_failed = []
            restored_count = 0
            for dest in destination_files:
                original = db.get_original_source(dest)
                if original:
                    try:
                        shutil.move(dest, original)
                        restored_count += 1
                    except Exception:
                        restore_failed.append(dest)
                else:
                    restore_failed.append(dest)

            print(f"{restored_count} files restored successfully")
            for f in restore_failed:
                print(f"Failed to restore: {f}")
        




if __name__ == "__main__":
    asyncio.run(main())
