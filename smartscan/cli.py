#! /usr/bin/env python3

import os
import argparse
import asyncio
import chromadb
import subprocess

from smartscan.utils.file import load_dir_list, get_files_from_dirs, get_child_dirs,clear_prototype_files
from smartscan.constants import  SCAN_HISTORY_DB, DB_DIR, SMARTSCAN_CONFIG_PATH, MODEL_REGISTRY
from smartscan.organiser.analyser import FileAnalyser
from smartscan.organiser.classifier import FileClassifier
from smartscan.organiser.classifier_listener import FileClassifierListener
from smartscan.indexer.indexer import FileIndexer
from smartscan.indexer.indexer_listener import FileIndexerListener
from smartscan.data.scan_history import ScanHistoryDB, ScanHistoryFilterOpts
from smartscan.config import load_config
from smartscan.utils.embeddings import get_image_encoder, get_text_encoder

def existing_file(path):
    if not os.path.isfile(path):
        raise argparse.ArgumentTypeError(f"File not found: {path}")
    return path

def existing_dir(path: str) -> str:
    if not os.path.isdir(path):
        raise argparse.ArgumentTypeError(f"Invalid directory: {path}")
    return path

async def cli():
    config = load_config(SMARTSCAN_CONFIG_PATH)
    text_encoder_path = MODEL_REGISTRY[config.text_encoder_model]['path']
    image_encoder_path = MODEL_REGISTRY[config.image_encoder_model]['path']

    if not os.path.isfile(text_encoder_path):
        raise ValueError(f"Text encoder model not found: {text_encoder_path} ")

    if not os.path.isfile(image_encoder_path):
        raise ValueError(f"Image encoder model not found: {image_encoder_path}")


    parser = argparse.ArgumentParser(description="CLI tool for comparing text or image embeddings and scanning directories.")
    subparsers = parser.add_subparsers(dest='command', required=True)

    common_parent = argparse.ArgumentParser(add_help=False)
    common_parent.add_argument("--clear-dir-prototypes",nargs="+", type=existing_dir, metavar=("DIRLIST"),help="Clear all prototype embeddings for dirs")

    compare_parser = subparsers.add_parser("compare", help="Perform comparisons using different modes.", parents=[common_parent])
    compare_parser.add_argument("--n-frames",type=int, default=10,help="Number of frames to use when generating video embedding. Default is 10")
    compare_parser.add_argument("file", default=10, type=existing_file,help="Source file to compare to")
    compare_group = compare_parser.add_mutually_exclusive_group(required=True)
    compare_group.add_argument("target_file", nargs="?",type=existing_file, metavar=("TARGETFILE"),help="Target file to be compared to source file.")
    compare_group.add_argument("--dirs",nargs="+", type=existing_dir, metavar=("DIRLIST"),help="List of target directories to be compared to source file.")
    compare_group.add_argument("--dirlist-file", type=existing_file,metavar=("DIRLISTFILE"),help="File listing target directories to be compared to source file.")

    autosort_parser = subparsers.add_parser("autosort", help="Scan directories and auto organise files into directories.", parents=[common_parent])
    autosort_parser.add_argument("-t", "--threshold",type=float,default=config.similarity_threshold,help="Similarity threshold for the scans. Default is 0.7.")
    autosort_parser.add_argument("--n-frames",type=int, default=10,help="Number of frames to use when generating video embedding. Default is 10")
    autosort_group = autosort_parser.add_mutually_exclusive_group(required=True)
    autosort_group.add_argument("dirlist_file", nargs="?", type=existing_file, metavar="DIRLISTFILE", help="File listing target directories to scan.")
    autosort_group.add_argument("--dirs",nargs='+', type=existing_dir,default=config.target_dirs, metavar="DIRLIST", help="List of directories to scan.")

    autosort_service_parser = subparsers.add_parser("autosort_service", help="Scan directories and auto organise files into directories.")
    autosort_service_parser.add_argument("--setup",action="store_true",help="Setup systemd background service to auto-organise files")
    autosort_service_parser.add_argument("--disable",action="store_true",help="Disable systemd background service to auto-organise files")
    autosort_service_parser.add_argument("--enable",action="store_true",help="Enable systemd background service to auto-organise files")
    autosort_service_parser.add_argument("--logs",action="store_true",help="View systemd service logs")

    index_parser = subparsers.add_parser("index", help="Index files from selected directories.")
    index_parser.add_argument("--n-frames",type=int, default=10,help="Number of frames to use when generating video embedding. Default is 10")
    index_group = index_parser.add_mutually_exclusive_group(required=True)
    index_group.add_argument("dirlist_file",nargs='?', type=existing_file,metavar="DIRLISTFILE", help="File listing target directories to index.")
    index_group.add_argument("--dirs",nargs='+', type=existing_dir,default=config.target_dirs, metavar="DIRLIST", help="List of directories to index.")

    restore_parser = subparsers.add_parser("restore", help="Restore files that were moved to their original location.")
    restore_parser.add_argument("--start-date",type=str,default=None,help="The start_date period of when to restore")
    restore_parser.add_argument("--end-date",type=str,default=None,help="The start_date period of when to restore")
    restore_group = restore_parser.add_mutually_exclusive_group(required=False)
    restore_group.add_argument("file",nargs='?',type=existing_file,metavar="FILETORESTORE",help="File to restore")
    restore_group.add_argument("--files",nargs='+',type=existing_file,metavar="FILESTORESTORE",help="List of files to restore.")

    args = parser.parse_args()

    client = chromadb.PersistentClient(path=DB_DIR)
    text_store = client.get_or_create_collection(
        name=f"{config.text_encoder_model}_text_collection",
        metadata={"description": "Collection for text documents"}
    )
    image_store = client.get_or_create_collection(
        name=f"{config.image_encoder_model}_image_collection",
        metadata={"description": "Collection for images"}
    ) 
    video_store = client.get_or_create_collection(
        name=f"{config.image_encoder_model}_video_collection",
        metadata={"description": "Collection for videos"}
    )

    if args.command == "compare":
        if args.clear_dir_prototypes:
            clear_prototype_files(args.clear_dir_prototypes)
            
        file_analyser = FileAnalyser(
            image_encoder=get_image_encoder(image_encoder_path),
            text_encoder=get_text_encoder(text_encoder_path),
            similarity_threshold=config.similarity_threshold,
            image_store=image_store,
            text_store=text_store,
            video_store=video_store,
            n_frames_limit=args.n_frames
        )

        file_analyser.image_encoder.init()
        file_analyser.text_encoder.init()

        if args.target_file:
            filepath1, filepath2 = args.file, args.target_file
            similarity_score = file_analyser.compare_files([filepath1, filepath2])
            print(f"File-to-file similarity: {similarity_score}")

        elif args.dirs or args.dirlist_file:
            source_filepath = args.file
            if args.dirlist_file:
                dirpaths = load_dir_list(args.dirlist_file)
            else:
                dirpaths = args.dirs
            best_dir, best_sim = file_analyser.compare_file_to_dirs(source_filepath, dirpaths)
            print(f"Best Directory: {best_dir} | Similarity: {best_sim}")

    elif args.command == "autosort":
        if args.clear_dir_prototypes:
            clear_prototype_files(args.clear_dir_prototypes)
            
        file_analyser = FileAnalyser(
            image_encoder=get_image_encoder(image_encoder_path),
            text_encoder=get_text_encoder(text_encoder_path),
            similarity_threshold=config.similarity_threshold,
            image_store=image_store,
            text_store=text_store,
            video_store=video_store,
            n_frames_limit=args.n_frames
        )

        file_analyser.similarity_threshold = args.threshold
       
        if args.dirlist_file:
            target_dirs = load_dir_list(args.dirlist_file)
        else:
            target_dirs = args.dirs

        file_analyser.image_encoder.init()
        file_analyser.text_encoder.init()

        file_scanner = FileClassifier(analyser=file_analyser, destination_dirs=get_child_dirs(target_dirs), db_path=SCAN_HISTORY_DB , listener = FileClassifierListener())
        allowed_exts = file_analyser.valid_img_exts + file_analyser.valid_txt_exts + file_analyser.valid_vid_exts
        files = get_files_from_dirs(target_dirs, allowed_exts=allowed_exts)
        result = await file_scanner.run(files)
        print(f"Scan results - files moved: {result.total_processed} | time elpased: {result.time_elapsed:.2f}s")

    elif args.command == "autosort_service":
        if args.setup:
            subprocess.run(['./systemd.sh', 'setup'])
        elif args.enable:
            subprocess.run(['./systemd.sh', 'enable'])
        elif args.disable:
            subprocess.run(['./systemd.sh', 'disable'])
        elif args.logs:
            subprocess.run(['./systemd.sh', 'logs'])
    
    elif args.command == "index":
        indexer = FileIndexer(
            image_encoder=get_image_encoder(image_encoder_path),
            text_encoder=get_text_encoder(text_encoder_path),
            image_store=image_store,
            text_store=text_store,
            video_store=video_store,
            listener = FileIndexerListener(),
            n_frames=args.n_frames
        )

        if args.dirlist_file:
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
            db.restore_files([args.file])
        elif args.files:
            db.restore_files(args.files)
        elif args.start_date or args.end_date:
            scans = db.get(filter_opts=ScanHistoryFilterOpts(start_date=args.start_date, end_date=args.end_date))
            if not scans:
                print(f"No scan history found matching dates")
                return
            destination_files = set([scan['destination_file'] for scan in scans])
            db.restore_files(destination_files)
    
if __name__ == "__main__":
    asyncio.run(cli())