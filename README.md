# SmartScan-Py

## Overview

SmartScan-Py is a Python tool for automated file management and search (desktop app integration coming soon) based on semantic similarity. It supports multiple embedding providers for generating embeddings and is the Python implementation of the [SmartScan Android app](https://github.com/dev-diaries41/smartscan).

## Features

* Supports text, image, and video files.
* Index files in target directories.
* Compare files or directories to determine semantic similarity.
* Automatically organize files based on similarity thresholds.
* Restore files to original location after unintended moves.
* Auto-organize files daily using systemd.

> **Caution:** Use high thresholds (e.g., 0.7+) for automatic organization to avoid undesired moves. The `restore` command can revert moves.

---

## Installation

### Prerequisites

* Python 3.10+

### Installation Steps

1. Clone the repository:

   ```bash
   git clone https://github.com/dev-diaries41/smartscan-py.git
   cd smartscan-py
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

SmartScan-Py supports four main commands: `compare`, `scan`, `index`, and `restore`.

### Supported File Types

* **Images:** `.png`, `.jpg`, `.jpeg`, `.bmp`, `.webp`
* **Text:** `.txt`, `.md`, `.rst`, `.html`, `.json`
* **Videos:** `.mp4`, `.mkv`, `.webm`

---

## Compare Command

Compare files or directories to measure semantic similarity.

```bash
smartscan compare [OPTIONS]
```

Modes:

1. **File-to-File**

   ```bash
   smartscan compare <FILEPATH> <TARGETFILE>
   ```

   Compare two files.

2a. **File-to-Multiple Directories (inline list)**

   ```bash
   smartscan compare <FILEPATH> --dirs DIR1 DIR2 DIR3
   ```

   Compare a file against multiple directories inline.

2b. **File-to-Multiple Directories (from file)**

   ```bash
   smartscan compare <FILEPATH> --dirlist-file <DIRLISTFILE>
   ```

   Compare a file against multiple directories listed in a file (one per line).

---

## Scan Command

Scan directories and automatically organize files.

```bash
smartscan scan [DIRLISTFILE or --dirs DIRS...] [-t THRESHOLD]
```

* `DIRLISTFILE` – text file listing directories to scan.
* `--dirs DIRS...` – alternatively, list directories directly.
* `-t, --threshold` – similarity threshold (default: `0.4`).

Example:

```bash
smartscan scan target_dirs.txt -t 0.8
smartscan scan --dirs /path/one /path/two -t 0.7
```

---

## Index Command

Index files from selected directories for faster future comparisons.

```bash
smartscan index [DIRLISTFILE or --dirs DIRS...] [--n-frames N]
```

* `--n-frames` – number of frames to use for video embeddings (default: `10`).

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

* `FILE` – single file to restore.
* `--files FILES...` – list of files to restore.
* `--start-date DATE` / `--end-date DATE` – restore files moved during a date range.

Example:

```bash
smartscan restore myfile.txt
smartscan restore --files file1.txt file2.txt
smartscan restore --start-date 2025-01-01 --end-date 2025-01-15
```

---

## Systemd Integration for Daily Scans

1. **Create SmartScan configuration** at `$HOME/.smartscan/smartscan.conf`:

```bash
TARGET_FILE=/path/to/target_dirs.txt
THRESHOLD=0.7
```

2. **Run setup script**:

```bash
chmod 777 setup_systemd.sh
./setup_systemd.sh
```

3. Adjust the schedule by editing `$HOME/.smartscan/systemd/smartscan.timer`.

---
