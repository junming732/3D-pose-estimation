# 3D Pose Estimation – Dataset Pipeline

This project runs a complete human pose estimation pipeline on pre-recorded videos from datasets such as [MPI-INF-3DHP dataset](https://vcai.mpi-inf.mpg.de/3dhp-dataset/). It detects 2-D human joint positions frame-by-frame, saves the results as compressed .npz files, creates optional overlay videos with skeletons, and can compute evaluation metrics (such as Mean Per-Joint Position Error, MPJPE) against provided ground truth.
It also includes a converter to transform the original annot.mat files from MPI-INF-3DHP into .npz format for easier evaluation.

## Features

- Run MediaPipe/TF-Lite 2-D pose detection on every frame of a dataset.
- Save predictions as compressed `.npz` files.
- (Optional) convert `annot.mat` files to `.npz` for evaluation.
- Compute 2-D MPJPE metrics against ground truth.
- Save visualization videos with skeleton overlay.

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/youruser/3D-pose-estimation.git
cd 3D-pose-estimation

# create a venv and install in editable mode
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e ".[dev]"
```

## Download the Dataset

```bash
# download MPI-INF-3DHP zip
wget -O mpi_inf_3dhp.zip https://vcai.mpi-inf.mpg.de/3dhp-dataset/mpi_inf_3dhp.zip
unzip mpi_inf_3dhp.zip -d data

# run the dataset’s own script to fetch the sequences
bash data/mpi_inf_3dhp/get_dataset.sh
```

You should end up with a structure like:

```
data/mpi_inf_3dhp/train/S1/Seq1/imageSequence/video_0.avi
data/mpi_inf_3dhp/train/S1/Seq1/annot.mat
…
```

## Convert Ground Truth

Convert the `annot.mat` files to `.npz` once. This will produce `data/gt_2d_all.npz` and `data/gt_3d_all.npz`:

```bash
python3 -m pose3d.convert_annot_to_npz     --config examples/config/dataset.yaml     --out-prefix data/gt
```

Then edit `examples/config/dataset.yaml` so `gt_2d_npz` and `gt_3d_npz` point to those files:

```yaml
gt_2d_npz: "data/gt_2d_all.npz"
gt_3d_npz: "data/gt_3d_all.npz"
```

## Run the Pipeline

Run on all matching videos:

```bash
python3 -m pose3d.dataset_runner     --config examples/config/dataset.yaml     --save-vis --eval
```

Run on just one file for testing:

```bash
python3 -m pose3d.dataset_runner     --config examples/config/dataset.yaml     --save-vis --eval --limit 1
```

Outputs go to `outputs/mpi3dhp/preds` (keypoints) and `outputs/mpi3dhp/vis` (overlay videos).

## Retrieve Results from Server

Copy the output videos back to your local machine:

```bash
scp -r ubuntu@your-server:/home/ubuntu/3D-pose-estimation/outputs/mpi3dhp/vis .
```

Or host a simple HTTP server to download:

```bash
python3 -m http.server 8080
```

Then open `http://<server>:8080/outputs/mpi3dhp/vis/` in your browser.


