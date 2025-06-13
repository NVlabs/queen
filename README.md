# QUEEN: QUantized Efficient ENcoding of Dynamic Gaussians for Streaming Free-viewpoint Videos

**QUEEN** (**QU**antized **E**fficient **EN**coding) is a novel framework for efficient, streamable free-viewpoint video (FVV) representation using dynamic 3D Gaussians. QUEEN enables high-quality dynamic scene capture, drastically reduces model size (to just 0.7 MB per frame), trains in under 5 seconds per frame, and achieves real-time rendering at ~350 FPS.

This repository contains the official implementation for QUEEN, as introduced in the NeurIPS 2024 paper:

[**QUEEN: QUantized Efficient ENcoding for Streaming Free-viewpoint Videos**](https://arxiv.org/abs/2412.04469)
[*Sharath Girish*](https://sharath-girish.github.io/),
[*Tianye Li**](https://tianyeli.github.io/),
[*Amrita Mazumdar**](https://amritamaz.net/),
[*Abhinav Shrivastava*](https://www.cs.umd.edu/~abhinav/),
[*David Luebke*](https://luebke.us/),
[*Shalini De Mello*](https://research.nvidia.com/person/shalini-de-mello)
NeurIPS 2024




For business inquiries, please visit our website and submit the form: [NVIDIA Research Licensing](https://www.nvidia.com/en-us/research/inquiries/).

---

## Links

- [Project Page](https://research.nvidia.com/labs/amri/projects/queen/)
- [arXiv/Preprint](https://arxiv.org/abs/2412.04469)
- [QUEEN GTC 2025 Demo Homepage](https://research.nvidia.com/labs/amri/projects/gtc2025-immersive-experiences/)

---

## 🔥 News
- 2025/06: Initial code release!
- 2025/03: We showed a [live demo of QUEEN](https://research.nvidia.com/labs/amri/projects/gtc2025-immersive-experiences/) in VR and light field displays at GTC San Jose 2025!
- 2024/12: [QUEEN](https://research.nvidia.com/labs/amri/projects/queen/) was published at NeurIPS 2024.

## Contents

- [🔥 News](#-news)
- [Contents](#contents)
- [🔧 Environment Setup](#environment-setup)
- [Data Preparation](#data-preparation)
- [💻 Training](#-training)
- [🎥 Rendering and Evaluation](#-rendering-and-evaluation)
  - [Rendering from Dense 3DGS](#rendering-from-dense-3dgs)
  - [Rendering from Compressed](#rendering-from-compressed)
- [📋 Pre-trained Models](#-pre-trained-models)
- [🎓 Citations](#-citations)
- [🙏 Acknowledgements](#-acknowledgements)


---

## Code Setup

### 0. Dependencies

We have only tested on Linux environments with CUDA 11.8+ compatible systems. 

### 1. Clone and Setup Repo

Our software has some submodules, please clone the repo recursively. 

```bash
git clone --recurse-submodules git@github.com:NVlabs/queen.git queen
cd queen
# set up some relevant directories
mkdir data
mkdir logs
mkdir output
```

### 2. Environment Setup

We suggest using the Dockerfile to reproduce the environment. We also provide a conda environment, b

#### Dockerfile

Please use the [Dockerfile](Dockerfile) for training within a working container. 

#### Conda Environment

```bash
# create the conda environment
conda env create -f environment.yml
conda activate queen

# install submodules
pip install ./submodules/simple-knn
pip install ./submodules/diff-gaussian-rasterization
pip install ./submodules/gaussian-rasterization-grad

# manually correct bug in timm package
# as in https://github.com/huggingface/pytorch-image-models/issues/1530#issuecomment-2084575852 
cp ./maxxvit.py /root/miniconda3/lib/python3.11/site-packages/timm/models/maxxvit.py
```

#### Manually patch `maxxvit.py` in `timm`

This repo uses an older version of timm which requires a patched version of `maxxvit.py`, following [this issue](https://github.com/huggingface/pytorch-image-models/issues/1530#issuecomment-2084575852).

We include this patched file in the repo (`maxxvit.py`). You’ll need to overwrite the existing file in your environment’s `timm` installation. This step is already included in the Dockerfile and Conda install instructions, but explained in further detail here. 

Python version and Conda environment paths can vary, so you must locate the correct destination and copy the file to that location:

```
python -c "import timm; print(timm.__file__)"
cp maxxvit.py /path/to/timm/models/maxxvit.py
```

For example, if you’re using Python 3.12 in a Conda env named `queen`:

```
cp maxxvit.py ~/miniconda3/envs/queen/lib/python3.12/site-packages/timm/models/maxxvit.py
```

### 3. Download weights for MiDaS

For MiDaS, please download their pretrained weights `dpt_beit_large_512.pt` from [their official repo](https://github.com/isl-org/MiDaS). We tested with the V3.1 release.

```
wget -P MiDaS/weights https://github.com/isl-org/MiDaS/releases/download/v3_1/dpt_beit_large_512.pt
```

---

## Data Preparation

We assume datasets are organized as follows:

### DyNeRF
```
| --- data
|   | [dataset_directory]
│     | [scene_name] 
│   	  | cam01
|            | images
|     		  | ---0000.png
│     		  | --- 0001.png
│     		  | --- ...
│   	  | cam02
|            | images
│     		  | --- 0000.png
│     		  | --- 0001.png
│     		  | --- ...
│   	  | ...
│   	  | sparse_
│     		  | --- cameras.bin
│     		  | --- images.bin
│     		  | --- ...
│   	  | points3D_downsample2.ply
│   	  | poses_bounds.npy
```

To generate the points3D_downsample2.ply , please use the [multipleviewprogress.sh](https://github.com/hustvl/4DGaussians/blob/master/multipleviewprogress.sh) script from [4DGaussians](https://github.com/hustvl/4DGaussians).

### Google Immersive

```
| --- data
|   | [dataset_directory]
│     | [scene_name] 
│   	  | camera_0001
|            | images_scaled_2
|     		  | ---0000.png
│     		  | --- 0001.png
│     		  | --- ...
│   	  | camera_0001
|            | images_scaled_2
│     		 | --- 0000.png
│     		  | --- 0001.png
│     		  | --- ...
│   	  | ...
│   	  | colmap
|            | sparse
|                | 0  
|     		  | --- points3d.bin
│     		  | --- cameras.bin
│     		  | --- images.bin
```

If you have a dataset of multi-view videos, please follow the steps to produce the required data:

1. organize the videos into folders of images for each camera view
2. follow the instructions in the [gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting?tab=readme-ov-file#processing-your-own-scenes) codebase to create the COLMAP camera data at the `sparse_` directory

For more information on how datasets are loaded, please see `scene/dataset_readers.py`.

---

## 💻 Training

You can train a scene by running:

```bash
python train.py --config [config_path] -s [source_path] -m [output_name]
```

For example:
```bash
python train.py --config configs/dynerf.yaml -s data/dynerf/coffee_martini -m ./output/coffee_martini_trained
```

The training script builds on the original 3DGS training script, and, as such, shares many of the same command line arguments. We add new arguments to control compression hyperparameters and 3DGS training hyperparameters for the initial frame and residual frames. 

Please see specific configuration files in `configs` for examples, and `arguments/__init__.py` for the full list of arguments.

<details>
<summary><span style="font-weight: bold;">Useful Command Line Arguments for train.py</span></summary>

  #### --source_path / -s
  Path to the source directory containing a COLMAP or Synthetic NeRF data set.
  #### --model_path / -m 
  Path where the trained model should be stored (```output/<random>``` by default).
  #### --white_background / -w
  Add this flag to use white background instead of black (default), e.g., for evaluation of NeRF Synthetic dataset.
  #### --sh_degree
  Order of spherical harmonics to be used (no larger than 3). ```3``` by default.
  #### --max_frames
  Maximum number of frames to process, ```300``` by default.
  #### --log_images
  Flag to save rendered images during training.
  #### --log_ply
  Flag to save point cloud in PLY format during training.
  #### --log_compressed
  Flag to save compressed model during training.
  #### --save_format
  Format to save the model in, ```ply``` by default.
  #### --use_xyz_legacy
  Flag to use the ```_xyz``` implementation to reproduce the paper results. Note that using the legacy implementation with ```--log_compressed``` is unsupported. 

</details>
<br>


---

## 🎥 Rendering and Evaluation

### Rendering from Dense 3DGS

```bash
python render.py -s <path to scene> -m <path to trained model> # Generate renderings
python metrics.py -m <path to trained model> # Compute error metrics on renderings
```

We also provide a script to render spiral viewpoints.
```bash
python render_fvv.py  -s <path to scene> -m <path to trained model> # Generate renderings
```

Example usage:
```
# Train coffee-martini scene
python train.py --config configs/dynerf.yaml --log_images --log_ply -s data/dynerf/coffee_martini -m ./output/coffee_martini_trained 

# Compute metrics
python metrics_video.py -m ./output/coffee_martini_trained 

# Render static camera viewpoints and spiral
python render.py -s data/dynerf/coffee_martini -m ./output/coffee_martini_trained
python render_fvv.py --config configs/dynerf.yaml  -s data/dynerf/coffee_martini -m ./output/coffee_martini_trained
```

### Rendering from Compressed

After training with `--log_compressed` to save compressed scenes, we can decode and render frames.
```bash
python render_fvv_compressed.py --config configs/dynerf.yaml [any additional config flags] -s <path to scene> -m <path to trained model> # Generate renderings
```

If you provided any additional command-line arguments during training to adjust the training configuration, you may need to add them to properly generate the renderings, as well.

---

## 📋 Pre-trained Models

We provide pre-trained models for the [N3DV dataset](https://github.com/facebookresearch/Neural_3D_Video). The provided models are subject to the Creative Commons — Attribution-NonCommercial-ShareAlike 4.0 International — CC BY-NC-SA 4.0 License terms. The download links for these models are provided in the table below, or on [the Models documentation](docs/models.md). Please use the ["render from compressed"](#rendering-from-compressed) instructions to render from the compressed pkl's. 

Note: we provide two versions of our models -- "NeurIPS24" and "Compressed." The "NeurIPS24" version includes only the dense .ply files corresponding to the training runs reported in the paper, while the "Compressed" version contains both dense .ply files and compressed .pkl representations. We identified and corrected a bug before code release that affected the rendering from the compressed .pkl files. This fix has resulted in a slight change in the results (we include a training flag, ```--use_xyz_legacy``` to train with or without the fix).

### NeurIPS24 -- Paper Models [(download)](https://github.com/NVlabs/queen/releases/tag/v1.0-neurips24)
| Scene | PSNR (dB)↑ | SSIM ↑ | LPIPS ↓ | 
|:-----:|:----------:|:------:|:-------:|
| coffee-martini | 28.31 | 0.916 | 0.155 |
| cook-spinach | 33.31 | 0.955 | 0.134 | 
| cut-roasted-beef | 33.64 | 0.958 | 0.132 | 
| sear-steak | 33.95 | 0.962 | 0.125 | 
| flame-steak | 34.16 | 0.962 | 0.125 | 
| flame-salmon | 29.17| 0.923 | 0.144 | 

### Compressed -- Corrected Compressed Models  [(download)](https://github.com/NVlabs/queen/releases/tag/v1.0-compressed)

| Scene | PSNR (dB)↑ | SSIM ↑ | LPIPS ↓ |  
|:-----:|:----------:|:------:|:-------:|
| coffee-martini | 28.22 | 0.915 | 0.156 | 
| cook-spinach | 33.33 | 0.956 | 0.134 |
| cut-roasted-beef | 33.49 | 0.958 | 0.133 |
| sear-steak | 33.94 | 0.962 | 0.126 |
| flame-steak | 34.17 | 0.962 | 0.126 |
| flame-salmon | 28.93 | 0.922 | 0.145 |

---

## 🎓 Citation

If you find QUEEN useful in your research, please cite:

```bibtex
@inproceedings{
girish2024queen,
  title={{QUEEN}: {QU}antized Efficient {EN}coding for Streaming Free-viewpoint Videos},
  author={Sharath Girish and Tianye Li and Amrita Mazumdar and Abhinav Shrivastava and David Luebke and Shalini De Mello},
  booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
  year={2024},
  url={https://openreview.net/forum?id=7xhwE7VH4S}
}
```

---

## 🙏 Acknowledgements

QUEEN builds upon the original [gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting) codebase, and uses the pretrained [MiDaS model](https://github.com/isl-org/MiDaS) for depth estimation. We thank the authors for their contributions.

---
