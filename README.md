Image Sorter by Visual Similarity

Description
This project is a Python script for sorting images based on visual similarity using pre-trained neural network models from torchvision. It extracts features from images, builds a graph of similarities, computes a Minimum Spanning Tree (MST), and traverses it to determine an optimal order. The sorted images are copied to an output folder with renamed files for easy sequential viewing.
Key features:

Supports various image formats: .jpg, .jpeg, .png, .bmp, .webp, .jfif.
Uses models like regnet_y_16gf (default), mobilenet_v3_small, etc., for feature extraction.
Caches features in a SQLite database to avoid reprocessing.
Optional "more_scan" mode for better handling of non-square images.
Efficient for large datasets (millions of images) with batch processing and FAISS for nearest neighbors.

Author: DZgas (@my_content)
Installation

Clone the repository:
git clone https://github.com/your-repo/image-sorter.git
cd image-sorter


Install dependencies:
pip install numpy tqdm scipy torch torchvision Pillow faiss-cpu


For GPU support: Install faiss-gpu and ensure CUDA is set up.
Note: No internet access required after initial model weights download.



Usage
Run the script with:
python sorter.py [arguments]

Command-Line Arguments

-m <model_name>: Model for feature extraction (default: regnet_y_16gf). Choices: mobilenet_v3_small, mobilenet_v3_large, convnext_small, regnet_y_16gf, regnet_y_32gf, regnet_y_128gf.
--more_scan: Enable advanced scanning mode (slower but more accurate for elongated images).
-i <input_folder>: Folder with source images (default: input_images).
-o <output_folder>: Folder for sorted images (default: sorted_images).
--cpu: Force CPU usage (even if GPU is available).

Example:
python sorter.py -m convnext_small -i my_images -o sorted_output --more_scan

How It Works

Feature Extraction: Loads a pre-trained model and extracts feature vectors from images. Caches them in a SQLite DB (features_db_<model>.sqlite).
Graph Construction: Uses FAISS to find nearest neighbors (Euclidean distance), builds a symmetric graph.
MST and Traversal: Computes MST, performs optimized DFS traversal (with lookahead for better paths).
Optimization: Applies 2-opt on path blocks for refinement.
Copying: Copies images to output folder with names like 0000_original.jpg.


If no new images, it reuses cached features.
Output files are copied (originals unchanged).

Requirements

Python 3.8+
Libraries: See Installation.
Hardware: GPU recommended for large datasets (use --cpu otherwise).
Disk space: For caching features and output copies.

Examples

Basic run:

Place images in input_images.
Run python sorter.py.
Sorted copies appear in sorted_images.


Advanced:

python sorter.py -m mobilenet_v3_small --cpu -i photos -o sorted_photos
Uses lightweight model on CPU.





Performance Tips

For millions of images: Use mobilenet_v3_small (fast but less accurate).
GPU accelerates feature extraction and FAISS.
"More scan" mode: Analyzes multiple crops for better similarity detection.

Limitations

Models are pre-trained on ImageNet; may not handle domain-specific images perfectly.
No real-time processing; batch-oriented.
Assumes images are RGB; converts if needed.

License
MIT License. See LICENSE for details.
Contributing
Pull requests welcome! For issues, open a ticket on GitHub.
