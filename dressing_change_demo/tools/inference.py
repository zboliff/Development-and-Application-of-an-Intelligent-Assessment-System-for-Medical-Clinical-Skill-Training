import argparse
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from mmengine.config import Config
from mmengine.runner import Runner
from mmaction.apis import init_recognizer
from mmaction.utils import register_all_modules
from mmcv.transforms import Compose

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.dataset import DatasetZelda, LoadVideoDir

def parse_args():
    parser = argparse.ArgumentParser(description='Inference for dressing change video understanding')
    parser.add_argument('--config', default='configs/tsm_config.py', help='config file path')
    parser.add_argument('--checkpoint', default='work_dirs/demo/checkpoint_best_acc.pth.tar', help='checkpoint file')
    parser.add_argument('--video', required=True, help='video file path or directory of frames')
    parser.add_argument('--data_root', default='extracted_data3_p', help='root directory for support set data')
    parser.add_argument('--device', default='cuda:1', help='device used for inference')
    parser.add_argument('--action_class', default='6', help='action class to train on')
    return parser.parse_args()

def extract_features(model, data_loader, device):
    model.eval()
    features = []
    labels = []
    filenames = []
    
    avg_pool = nn.AdaptiveAvgPool2d((1, 1))
    
    with torch.no_grad():
        for data_batch in data_loader:
            # Move data to device
            data = model.data_preprocessor(data_batch, training=False)
            
            # Extract features
            # model(**data, mode='tensor') returns the feature map
            prediction = model(**data, mode='tensor')
            
            # Pool and normalize
            prediction = avg_pool(prediction)
            prediction = prediction.view(len(data_batch['data_samples']), -1)
            prediction = F.normalize(prediction, dim=1)
            
            features.append(prediction.cpu())
            
            for item in data_batch['data_samples']:
                labels.append(item.gt_label.item())
                # Handle filename/path storage if needed
                # filenames.append(item.img_path if hasattr(item, 'img_path') else str(item))

    features = torch.cat(features, dim=0)
    labels = torch.tensor(labels)
    return features, labels

def main():
    args = parse_args()
    
    register_all_modules(init_default_scope=True)
    
    # 1. Build Model
    print(f"Building model from {args.config}...")
    try:
        model = init_recognizer(args.config, args.checkpoint, device=args.device)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Trying to load checkpoint manually if strict loading failed...")
        # Fallback or custom loading if needed, but init_recognizer is usually robust
        raise e

    # 2. Build Support Loader
    print("Building support set loader...")
    
    # Reconstruct the pipeline config for support set (same as val/test)
    file_client_args = dict(io_backend='disk')
    val_pipeline_cfg = [
        dict(type='LoadVideoDir', **file_client_args),
        dict(
            type='SampleFrames',
            clip_len=1,
            frame_interval=1,
            num_clips=16,
            test_mode=True),
        dict(type='RawFrameDecode', io_backend='disk', decoding_backend='cv2'),
        dict(type='Resize', scale=(-1, 256)),
        dict(type='CenterCrop', crop_size=224),
        dict(type='FormatShape', input_format='NCHW'),
        dict(type='PackActionInputs')
    ]
    
    dataset_type = 'DatasetZelda'
    support_dataloader_cfg = dict(
        batch_size=16,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        sampler=dict(type='DefaultSampler', shuffle=False), # No shuffle for deterministic features
        dataset=dict(
            type=dataset_type,
            data_root=args.data_root,
            action_path=args.action_class,
            pre='support',
            pipeline=val_pipeline_cfg,
            test_mode=True))
            
    support_data_loader = Runner.build_dataloader(dataloader=support_dataloader_cfg)
    
    # 3. Extract Support Features
    print("Extracting support set features...")
    support_features, support_labels = extract_features(model, support_data_loader, args.device)
    print(f"Support features shape: {support_features.shape}")
    
    # 4. Process Query Video
    print(f"Processing query video: {args.video}")
    
    # Construct pipeline for query video
    # Check if input is a file or directory
    is_file = os.path.isfile(args.video)
    
    if is_file:
        # Use Decord for video files
        print("Input is a video file. Using DecordInit.")
        test_pipeline_cfg = [
            dict(type='DecordInit', io_backend='disk'),
            dict(
                type='SampleFrames',
                clip_len=1,
                frame_interval=1,
                num_clips=16,
                test_mode=True),
            dict(type='DecordDecode'),
            dict(type='Resize', scale=(-1, 256)),
            dict(type='CenterCrop', crop_size=224),
            dict(type='FormatShape', input_format='NCHW'),
            dict(type='PackActionInputs')
        ]
    else:
        # Use LoadVideoDir for frame directories
        print("Input is a directory. Using LoadVideoDir.")
        test_pipeline_cfg = [
            dict(type='LoadVideoDir', **file_client_args),
            dict(
                type='SampleFrames',
                clip_len=1,
                frame_interval=1,
                num_clips=16,
                test_mode=True),
            dict(type='RawFrameDecode', io_backend='disk', decoding_backend='cv2'),
            dict(type='Resize', scale=(-1, 256)),
            dict(type='CenterCrop', crop_size=224),
            dict(type='FormatShape', input_format='NCHW'),
            dict(type='PackActionInputs')
        ]

    pipeline = Compose(test_pipeline_cfg)
    
    # Prepare data dict
    data = dict(filename=args.video, label=-1, start_index=0, modality='RGB')
    
    # Run pipeline
    data = pipeline(data)
    
    # Collate (add batch dimension)
    from mmengine.dataset import pseudo_collate
    data_batch = pseudo_collate([data])
    
    # 5. Extract Query Features
    print("Extracting query features...")
    model.eval()
    avg_pool = nn.AdaptiveAvgPool2d((1, 1))
    
    with torch.no_grad():
        data_batch = model.data_preprocessor(data_batch, training=False)
        query_feature = model(**data_batch, mode='tensor')
        query_feature = avg_pool(query_feature)
        query_feature = query_feature.view(1, -1)
        query_feature = F.normalize(query_feature, dim=1)
        query_feature = query_feature.cpu()

    # 6. Compute Similarity & Predict
    print("Computing similarity...")
    
    # Cosine similarity: (1, D) @ (N, D).T -> (1, N)
    similarity = torch.matmul(query_feature, support_features.T)
    
    # Find top matches
    topk_scores, topk_indices = torch.topk(similarity, k=min(5, len(support_labels)), dim=1)
    
    print("\nTop matches:")
    for i in range(len(topk_indices[0])):
        idx = topk_indices[0][i].item()
        score = topk_scores[0][i].item()
        label = support_labels[idx].item()
        print(f"Rank {i+1}: Label {label}, Score {score:.4f}")
        
    # Aggregate scores by class (Max score per class as in val_model.py)
    class_scores = {}
    unique_labels = torch.unique(support_labels)
    
    for label in unique_labels:
        label = label.item()
        # Find indices of this class in support set
        indices = (support_labels == label).nonzero(as_tuple=True)[0]
        # Get scores for this class
        scores = similarity[0, indices]
        # Max score
        max_score = torch.max(scores).item()
        class_scores[label] = max_score
        
    # Predict
    predicted_label = max(class_scores, key=class_scores.get)
    print(f"\nPredicted Label: {predicted_label} (Score: {class_scores[predicted_label]:.4f})")

if __name__ == '__main__':
    main()
