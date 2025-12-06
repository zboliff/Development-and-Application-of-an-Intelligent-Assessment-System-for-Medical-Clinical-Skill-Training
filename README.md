# Development and Application of an Intelligent Assessment System for Medical Clinical Skill Training

This project demonstrates a video understanding model for dressing change surgery, based on TSM (Temporal Shift Module) and contrastive learning.

## Directory Structure

```
dressing_change_demo/
├── configs/
│   └── tsm_config.py       # Model configuration
├── tools/
│   ├── train.py            # Training script
│   └── inference.py        # Inference script
├── utils/
│   ├── dataset.py          # Custom dataset loading
│   └── losses.py           # Contrastive loss functions
└── README.md
```

## Prerequisites & Setup

### 1. Install Dependencies

Ensure you have Python 3.8+ and PyTorch installed. Then install MMAction2 and its dependencies:

```bash
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"
mim install "mmaction2>=1.0.0"
pip install pandas numpy
```

### 2. Install MMAction2 Repository

The configuration files rely on the MMAction2 repository structure.

@misc{2020mmaction2,
    title={OpenMMLab's Next Generation Video Understanding Toolbox and Benchmark},
    author={MMAction2 Contributors},
    howpublished = {\url{https://github.com/open-mmlab/mmaction2}},
    year={2020}
}

### 3. Download Pretrained Weights

Download the TSM checkpoint required for training/inference and place it in the `dressing_change_demo` directory:

```bash
mim download mmaction --config tsm_imagenet-pretrained-r50_8xb16-1x1x16-50e_sthv2-rgb --dest .
```
*Note: Ensure the downloaded filename matches the default in `train.py` (e.g., `tsm_imagenet-pretrained-r50_8xb16-1x1x16-50e_sthv2-rgb_20230317-ec6696ad.pth`) or pass the path via `--checkpoint`.*

## Data Preparation

The code expects the following data structure:

```
data_root/
├── summary.xlsx            # Excel file with video metadata
├── dataset_splits_6/       # Split files for action '6'
│   ├── train_set.csv
│   ├── support_set.csv
│   └── test_set.csv
└── 6/                      # Video data (extracted frames or video files)
    ├── video_1/
    ├── video_2/
    ...
```

## Usage

### Training

To train the model:

```bash
python dressing_change_demo/tools/train.py --config /path/to/configs/xx.py --data_root /path/to/data --work_dir work_dirs/demo --action_class 6
```

Arguments:
- `--config`: Path to the configuration file.
- `--checkpoint`: Path to the pretrained checkpoint (optional).
- `--data_root`: Root directory of the dataset.
- `--work_dir`: Directory to save logs and checkpoints.
- `--epochs`: Number of training epochs.
- `--batch_size`: Batch size.
- `--action_class`: The action class to train on (e.g., '6', '3.1', '3.2', '5.2'). Default is '6'.

### Inference

To run inference on a video (or frame directory):

```bash
python dressing_change_demo/tools/inference.py --config /path/to/configs/xx.py --checkpoint work_dirs/demo/checkpoint_best_acc.pth.tar --video /path/to/video_or_frames
```

## Model Details

The model uses a TSM backbone with a contrastive learning objective. It utilizes a support set to guide the training, minimizing intra-class distance and maximizing inter-class distance using `intra_contrastive_loss` and `cross_contrastive_loss`.
