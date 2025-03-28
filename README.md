# SmartScan-CLI

SmartScan-CLI is a command-line tool for Linux systems with systemd support. It uses OpenAI's CLIP model to compare vector embeddings and measure similarity between files and directories. It can be used for manual comparisons or automated file organization. It is the CLI version of the [SmartScan Android app](https://github.com/dev-diaries41/smartscan).

## Features
- Compare a file to a directory of files to find the most similar match.
- Compare two files to determine their similarity.
- Compare a file to multiple directories using a list file.
- Scan and automatically organize files into destination directories based on similarity thresholds, monitoring target directories for changes.

---

## Installation  

### Prerequisites  
- Python 3.10  

### Installation Steps  
1. Clone the repository:  
   ```sh
   git clone <repo_url>
   cd smartscan-cli
   ```  
2. Set executable permissions for the install script:  
   ```sh
   chmod 777 install.sh
   ```  
3. Run the installation script:  
   ```sh
   ./install.sh
   ```  
   This will install the required dependencies, create a virtual environment, and set up the necessary directories. All relevant configuration and data files will be stored in the `$HOME/.smartscan` directory.

---


## Usage

SmartScan uses two main commands: **compare** and **scan**. The **compare** command provides different modes for comparing files and directories, while the **scan** command is used for automatically organizing files based on a similarity threshold.

### Supported file types

Image: '.png', '.jpg', '.jpeg', '.bmp', '.gif'
Text: '.txt', '.md', '.rst', '.html', '.json'

### Compare Command

The `compare` command offers three modes, determined by the flags used:

#### File-to-File Comparison
Compare two files to measure their similarity.
Flag: `-f or --file`

```sh
smartscan compare -f <FILEPATH1> <FILEPATH2>
```
Example:
```sh
smartscan compare -f doc1.txt doc2.txt
```

#### File-to-Directory Comparison
Compare a single file against a directory of files to determine a similarity between the file and the directory's contents.
Flag: `-d or --dir`

```sh
smartscan compare -d <FILEPATH> <DIRPATH>
```
Example:
```sh
smartscan compare -d sample.txt documents/
```

#### File-to-Multiple Directories Comparison
Compare a file against multiple directories specified in a list file.
Flag: `-l or --dirs`

```sh
smartscan compare -l <FILEPATH> <DIRLISTFILE>
```
Example:
```sh
smartscan compare -l sample.txt dir_list.txt
```
Where `dir_list.txt` contains a list of directories (one per line).


### Scan Command

The `scan` command automatically scans target directories and moves files to the best matching destination directory if the similarity meets a specified threshold (default is 0.7).

```sh
smartscan scan <TARGET_FILE> <DESTINATION_FILE> [-t THRESHOLD]
```
Example:
```sh
smartscan scan target_dirs.txt destination_dirs.txt -t 0.8
```
- **TARGET_FILE**: A file listing the target directories to monitor for changes.
- **DESTINATION_FILE**: A file listing the directories where files should be moved based on similarity.
- **-t, --threshold**: (Optional) Set a custom similarity threshold for file organization. Defaults to 0.7 if not provided.


### Onnx quantization

Instead of directly using the model returned from `open_clip.create_model_and_transforms()` an onnx quantized model is used instead to improve performance and reduce memory usage. If for any reason you want to use the original model directly uncomment the following line in `main.py` and update accorindly:

```bash
    # Uncomment this line if you do not wish to use the quantized onnx model then add model to all the 4 function below using model=model
 # model, _, _ = open_clip.create_model_and_transforms(MODEL_NAME, pretrained=PRETRAINED)
```
---

## Systemd Integration for Daily Scans

To make it easier to run scans daily, you can set up a systemd timer to execute SmartScan automatically. Follow these steps to enable it:

1. **Create the SmartScan Configuration File**:
   The SmartScan configuration file (`smartscan.conf`) is essential for defining the directories to be monitored and where files should be moved. You need to manually create this file in the `$HOME/.smartscan` directory.

   Example of `smartscan.conf`:

   ```bash
   TARGET_FILE=/path/to/target_dirs.txt
   DESTINATION_FILE=/path/to/destination_dirs.txt
   THRESHOLD=0.7 #optional
   ```

2. **Setup systemd using `setup_systemd.sh`**:
   The `setup_systemd.sh` script will automatically move the necessary systemd files to the appropriate system directories and enable the timer to run SmartScan daily. To do this, simply run:

  ```sh
   chmod 777 setup_systemd.sh
   ```

   ```sh
   ./setup_systemd.sh
   ```

3. **After running the script**, your system will be set up to automatically run SmartScan scans daily. You can adjust the scan schedule by modifying the `smartscan.timer` file located in `$HOME/.smartscan/systemd/`.

---
