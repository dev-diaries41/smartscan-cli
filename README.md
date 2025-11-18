# SmartScan Server

## Overview

Standalone CLI tool and server to power the SmartScan Desktop app (coming soon), providing automated file management and semantic search functionality.

## Features

* Supports text, image, and video files.
* Index files in target directories.
* Compare files or directories to determine semantic similarity.
* Automatically organize files based on similarity thresholds.
* Restore files to original location after unintended moves.
* Auto-organize files daily using systemd (autosort).
* Supports multiple embedding providers.

---

## Installation

### Prerequisites

* Python 3.10+

### Installation Steps

1. Clone the repository:

   ```bash
   git clone https://github.com/smartscanapp/smartscan-server.git
   cd smartscan-server
   ```
2. Set executable permissions for the install script:

   ```bash
   chmod 777 install.sh
   ```
3. Run the installation script:

   ```bash
   ./install.sh
   ```

   This installs dependencies, creates a virtual environment, and sets up necessary directories under `$HOME/.smartscan`.

---

## Usage

> **Caution:** Use high thresholds (e.g., 0.7+) for `autosort` to avoid undesired moves. The `restore` command can revert moves.

Five main commands: `compare`, `autosort`, `autosort_service`, `index`, and `restore`.

### Supported File Types

* **Images:** `.png`, `.jpg`, `.jpeg`, `.bmp`, `.webp`
* **Text:** `.txt`, `.md`, `.rst`, `.html`, `.json`
* **Videos:** `.mp4`, `.mkv`, `.webm`

---

## Compare Command

Compare files or directories to measure semantic similarity.

```bash
smartscan compare [OPTIONS] <FILE> [TARGETFILE]
```

Options:

* `--dirs DIR1 DIR2 ...` – Compare file against multiple directories.
* `--dirlist-file FILE` – Compare file against directories listed in a file.
* `--n-frames N` – Number of frames for video embeddings (default 10).
* `--clear-dir-prototypes DIRS...` – Clear prototype embeddings for directories.

Example:

```bash
smartscan compare myfile.txt targetfile.txt
smartscan compare myfile.txt --dirs /dir1 /dir2
smartscan compare myfile.txt --dirlist-file target_dirs.txt
```

---

## Autosort Command

Scan directories and automatically organize files.

```bash
smartscan autosort [OPTIONS] DIRLISTFILE
```

Options:

* `--dirs DIR1 DIR2 ...` – List of directories to scan instead of using a file.
* `-t, --threshold FLOAT` – Similarity threshold (default: 0.7).
* `--n-frames N` – Number of frames for video embeddings (default: 10).
* `--clear-dir-prototypes DIRS...` – Clear prototype embeddings for directories.

Example:

```bash
smartscan autosort target_dirs.txt -t 0.8
smartscan autosort --dirs /path/one /path/two -t 0.7
```

---

## Autosort Service Command

Manage systemd background service for daily auto-organization:

```bash
smartscan autosort_service [--setup | --enable | --disable | --logs]
```

* `--setup` – Setup systemd service.
* `--enable` – Enable systemd service.
* `--disable` – Disable systemd service.
* `--logs` – View systemd logs.

---

## Index Command

Index files for faster future comparisons.

```bash
smartscan index [OPTIONS] DIRLISTFILE
```

Options:

* `--dirs DIR1 DIR2 ...` – List directories to index.
* `--n-frames N` – Number of frames for video embeddings (default 10).

Example:

```bash
smartscan index dir_list.txt
smartscan index --dirs /videos /images --n-frames 15
```

---

## Restore Command

Restore previously moved files.

```bash
smartscan restore [OPTIONS]
```

Options:

* `FILE` – Single file to restore.
* `--files FILE1 FILE2 ...` – Multiple files to restore.
* `--start-date DATE` / `--end-date DATE` – Restore files moved within a date range.

Example:

```bash
smartscan restore myfile.txt
smartscan restore --files file1.txt file2.txt
smartscan restore --start-date 2025-01-01 --end-date 2025-01-15
```

---

## Systemd Integration for Daily Scans

1. Create SmartScan configuration at `$HOME/.smartscan/smartscan.json`:

```bash
{
    "similarity_threshold": 0.7,
    "target_dirs": ["/path/target-dir-one", "/path/target-dir-two"],
    "image_encoder_model": "dino",
    "text_encoder_model": "minilm"
}
```

2. Run setup command:

```bash
smartscan autosort_service --setup 
```

3. Adjust the schedule by editing `$HOME/.smartscan/systemd/smartscan.timer`.
