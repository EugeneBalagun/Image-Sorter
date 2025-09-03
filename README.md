# 🖼️ Image Sorter by Visual Similarity

A Python script to **sort images based on visual similarity** using pre-trained neural network models from `torchvision`.  
It extracts features, builds a similarity graph, computes a **Minimum Spanning Tree (MST)**, and traverses it to determine an optimal order.  
Sorted images are copied to an output folder with renamed files for easy sequential viewing.

---
![Описание картинки](screens/1.jpg)


## 🚀 Key Features

- Supports image formats: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`, `.jfif`.  
- Uses models like `regnet_y_16gf` (default), `mobilenet_v3_small`, `convnext_small`, and more.  
- Caches features in a SQLite database to avoid reprocessing.  
- Optional `--more_scan` mode for better handling of non-square images.  
- Efficient for **large datasets** (millions of images) with batch processing and FAISS for nearest neighbors.

---

## 💻 Installation

1. Clone the repository:

```bash
git clone https://github.com/your-repo/image-sorter.git
cd image-sorter
```
Install dependencies:

```bash
Copy code
pip install numpy tqdm scipy torch torchvision Pillow faiss-cpu
```
For GPU support: install faiss-gpu and ensure CUDA is set up.
Note: No internet required after initial model weights download.

⚡ Usage
Run the script with:

```bash
Copy code
python sorter.py [arguments]
Command-Line Arguments
Argument	Description	Default
-m <model_name>	Model for feature extraction	regnet_y_16gf
--more_scan	Advanced scanning mode (slower, more accurate for elongated images)	-
-i <input_folder>	Folder with source images	input_images
-o <output_folder>	Folder for sorted images	sorted_images
--cpu	Force CPU usage	-
```

Example
```bash
Copy code
python sorter.py -m convnext_small -i my_images -o sorted_output --more_scan
```
🛠 How It Works
Feature Extraction: Loads a pre-trained model and extracts feature vectors from images.
Cached in a SQLite DB (features_db_<model>.sqlite).

Graph Construction: Uses FAISS to find nearest neighbors (Euclidean distance) and builds a symmetric graph.

MST and Traversal: Computes a Minimum Spanning Tree and performs optimized DFS traversal (with lookahead for better paths).

Optimization: Applies 2-opt on path blocks for refinement.

Copying: Copies images to output folder with names like 0000_original.jpg.

Original images remain unchanged.

Cached features reused if no new images are added.

🎯 Requirements
Python 3.8+

Libraries: see Installation

Hardware: GPU recommended for large datasets (use --cpu otherwise)

Disk space: required for caching features and output copies

🔧 Examples
Basic Run
Place images in input_images

Run:

```bash
Copy code
python sorter.py
```
Sorted copies appear in sorted_images.

Advanced Run
```bash
Copy code
python sorter.py -m mobilenet_v3_small --cpu -i photos -o sorted_photos
```
Uses lightweight model on CPU, suitable for large datasets without GPU.

⚡ Performance Tips
For millions of images: use mobilenet_v3_small (fast but less accurate)

GPU accelerates feature extraction and FAISS

--more_scan mode analyzes multiple crops for better similarity detection

⚠️ Limitations
Models are pre-trained on ImageNet; domain-specific images may not be perfect

No real-time processing; batch-oriented

Assumes images are RGB; converts if needed
