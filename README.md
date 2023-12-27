# Fluorescent Cell Image Segmentation Benchmark

This code repository accompanies the article titled "A Generative Benchmark for Evaluating the Performance of Fluorescent Cell Image Segmentation." The toolkit, outlined in this repository, serves two main purposes: it facilitates the generation of diverse cell contours using a StyleGAN2-based approach and enables realistic contour rendering through the utilization of Pix2PixHD.

- **Cell Contour Generation:** The toolkit employs StyleGAN2 to generate diverse and graded-density cell contours.
- **Contour Rendering:** Leveraging Pix2PixHD, the repository enables realistic rendering of cell contours.

## Requirements

- Linux and Windows are supported, but we recommend Linux for performance and compatibility reasons.
- 1–8 high-end NVIDIA GPUs with at least 12 GB of memory. 
- 64-bit Python 3.8 and PyTorch 1.12.0 (or later). See https://pytorch.org for PyTorch install instructions. CUDA toolkit 11.6 or later.
- GCC 7 or later (Linux) or Visual Studio (Windows) compilers. Recommended GCC version depends on CUDA version, see for example [CUDA 11.6 system requirements](https://docs.nvidia.com/cuda/archive/11.6.0/cuda-installation-guide-linux/index.html#system-requirements).
- Python libraries: see `environment.yml `for exact library dependencies. You can use the following commands with Miniconda3/Anaconda3 to create and activate your cell Python environment:
  - `conda env create -f environment.yml`
  - `conda activate cell`

## Getting Started

- Clone this repo:

```bash
git clone https://github.com/edwardcao3026/SegBenchmark.git
cd SegBenchmark
```

## 1.Cell Contour Generation

- Trained networks are stored as `*.pkl` files that can be referenced using local filenames.

- Generate diverse cell contours using the following command:

```python
python generate.py --outdir=./output_folder --seeds=number of seeds --network= path to netwrok 
```



## 2.Image Rendering

- Utilize `test.py` for realistic image rendering:

```bash

python test.py --name cell --label_nc 0 --no_instance --which_epoch 80 --how_many numbers to generate
```



## 3.Image Segmentation

At this stage, we will use the generated dataset to compare the performance of various segmentation algorithms. This process involves inputting the generated dataset into different segmentation algorithms and evaluating their performance and effectiveness. By comparing different algorithms, we can find the segmentation algorithm that is most suitable for a specific task, which can help us improve the accuracy and efficiency of segmentation in practical applications. 

Here are the specific steps for using three different segmentation methods in articles:

### 3.1 [CellPose](https://www.CellPose.org/) <a href="#refer-1">[1]</a>

#### 3.1.1 Run CellPose in GUI

```bash
# Install cellpose and the GUI dependencies from your base environment using the command
python -m pip install cellpose[gui]

# The quickest way to start is to open the GUI from a command line terminal.
python -m cellpose
```

- Load an image in the GUI (by dragging and dropping the image or by selecting "Load" from the File menu).
- Set the model: CellPose has two models, cytoplasm and nuclei, which correspond to the segmentation of cell cytoplasm and cell nuclei, respectively.
- Set the channels: Select the image channel to be segmented. If segmenting cytoplasm, select the green channel; if segmenting nuclei, select the red/blue channel. If there are both cytoplasm and nuclei in the image, set chan to the channel where cytoplasm appears, and set chan2 to the channel where nuclei appear. If segmenting cytoplasm but there is no nucleus, only set chan and leave chan2 as None.
- Click on the calibrate button to estimate the size of objects in the image. Alternatively, manually input the cell diameter for calibration. The estimated size will be reflected by a red circle in the lower left corner.
- Click on the "run segmentation" button to start the segmentation process. You can choose to display the segmentation mask by checking the "MASKS ON" option.

#### 3.1.2 Run CellPose in Terminal

The parameter inputs in the GUI interface can also be achieved through the terminal mode:

```bash
python -m cellpose --dir ~/images_cyto/test/ --pretrained_model cyto --chan 2 --chan2 3 --save_png
```

All parameters can be viewed using the help parameter:

```bash
python -m cellpose -h
```

#### 3.1.3 Run CellPose in Code

Similar to the previous two methods, CellPose can also be called directly in Python code for programming: 

```python
from cellpose import models
import skimage.io

model = models.Cellpose(gpu=False, model_type='cyto')

files = ['img0.tif', 'img1.tif']

imgs = [skimage.io.imread(f) for f in files]

masks, flows, styles, diams = model.eval(imgs, diameter=None, channels=[0,0],
                                         threshold=0.4, do_3D=False)
```

### 3.2 CellPofiler <a href="#refer-1">[2]</a>

[CellProfiler](https://cellprofiler.org/) is a free software developed by the Broad Institute of Harvard and MIT. It is designed to enable biologists to quantitatively measure phenotypes of thousands of images automatically, without the need for computer vision or programming training.

#### 3.2.1 How to use CellPofiler

Creating a specific pipeline for cell segmentation in CellProfiler involves a series of steps and modules tailored to this task. Here's a outline of what such a pipeline might look like:

1. **Load Images:**
   - Use the `Images` module to specify the images you want to analyze.

2. **Identify Primary Objects:**
   - Employ the `IdentifyPrimaryObjects` module. This is crucial for cell segmentation, as it identifies individual cells in your images. Common settings involve specifying the typical size of the cells and the intensity threshold for segmentation.

3. **Identify Secondary Objects:**
   - If your analysis requires identifying objects surrounding or attached to the primary objects (like cytoplasm around a nucleus), use `IdentifySecondaryObjects`. This module relies on the primary objects as a reference.

4. **Measure Object Size and Shape:**
   - The `MeasureObjectSizeShape` module measures various properties of the identified objects, such as area, perimeter, and form factor.

5. **Export Data:**
   - Finally, use `ExportToSpreadsheet` and `SaveImages` modules to export your results for further analysis.

### 3.3 DeepCell <a href="#refer-1">[3]</a>

Researchers have developed a deep learning segmentation algorithm, Mesmer, which consists of a ResNet50 backbone and a feature pyramid network. It automatically extracts key cell features, such as subcellular localization of protein signals, achieving human-level performance.

#### 3.3.1Run DeepCell in website

Visit the pre-trained deep learning models on [DeepCell.org](https://DeepCell.org/). This website allows you to easily upload example images, run them on available models, and download the results without any local installation required.

#### 3.3.2 Run DeepCell in Docker

1. **Install DeepCell with pip:**
   ```python
   pip install deepcell
   ```

2. **Using DeepCell with Docker:**
   - If you have a GPU, ensure you have CUDA and Docker installed.
   - Run the Docker command to start a container with DeepCell installed:
     ```bash
     docker run --gpus '"device=0"' -it --rm -p 8888:8888 -v $PWD/notebooks:/notebooks -v $PWD/data:/data vanvalenlab/DeepCell-tf:latest-gpu
     ```
   - This command starts a Jupyter session and mounts data and notebook directories.

3. **Example Usage:**
   - The DeepCell documentation includes examples of training segmentation and tracking models.
   - You can find Python notebooks for these examples, illustrating how to use DeepCell for single-cell analysis.

For more detailed information and examples, you should refer to the [DeepCell documentation](https://DeepCell.readthedocs.io/en/master/). This resource provides comprehensive guidance on installing and using DeepCell, including example notebooks for various applications.

#### 3.3.3 Run DeepCell in Code

To run DeepCell in Python, you can follow this script:

```python
import os
import numpy as np
from skimage.io import imread, imsave
from DeepCell.applications import Mesmer

def binarize_image(image):
    """Binarize an image: set all non-zero pixels to 1."""
    return (image > 0).astype(np.uint8)

def process_directory(input_dir, output_dir, image_mpp=0.17):
    cytoplasm_path = os.path.join(output_dir, "cytoplasm")
    nucleus_path = os.path.join(output_dir,"nuclei")

    if not os.path.exists(cytoplasm_path):
        os.makedirs(cytoplasm_path)
    if not os.path.exists(nucleus_path):
        os.makedirs(nucleus_path)

    app = Mesmer()

    for filename in os.listdir(input_dir):
        if filename.endswith(".jpg"):
            filepath = os.path.join(input_dir, filename)
            
            im = imread(filepath)

            im = im[:, :, [1, 2]]

            im = np.expand_dims(im, axis=0)
            
            labeled_image = app.predict(im, compartment='both', image_mpp=image_mpp)
            
            binarized_cytoplasm = binarize_image(labeled_image[0, :, :, 0])
            binarized_nucleus = binarize_image(labeled_image[0, :, :, 1])

            base_filename = os.path.splitext(filename)[0] + ".png"

            cytoplasm_output_path = os.path.join(cytoplasm_path, base_filename)
            nucleus_output_path = os.path.join(nucleus_path, base_filename)
            imsave(cytoplasm_output_path, binarized_cytoplasm)
            imsave(nucleus_output_path, binarized_nucleus)


input_directory = 'your input path'
output_directory = 'your save path'
process_directory(input_directory, output_directory)
```



### 3.4 Evaluation Performance

After running the three segmentation methods as described above, you can proceed to analyze the segmentation results by running the `analysis.py` script. The results of the analysis will be saved as a CSV (Comma-Separated Values) file.

## Reference

<div id="refer-1"></div>
[1] Stringer, Carsen, et al. "Cellpose: a generalist algorithm for cellular segmentation." Nature methods 18.1 (2021): 100-106.

<div id="refer-2"></div>
[2] Carpenter, Anne E., et al. "CellProfiler: image analysis software for identifying and quantifying cell phenotypes." Genome biology 7 (2006): 1-11.

<div id="refer-3"></div>
[3] Greenwald, Noah F., et al. "Whole-cell segmentation of tissue images with human-level performance using large-scale data annotation and deep learning." Nature biotechnology 40.4 (2022): 555-565.
