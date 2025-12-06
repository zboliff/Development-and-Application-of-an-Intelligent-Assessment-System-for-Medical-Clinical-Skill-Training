import argparse
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import logging
import yaml
import shutil

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from mmaction.utils import register_all_modules
from mmaction.apis import init_recognizer
from mmengine.runner import Runner
from utils.dataset import DatasetZelda
from utils.losses import intra_contrastive_loss, cross_contrastive_loss

def parse_args():
    parser = argparse.ArgumentParser(description='Train dressing change video understanding model')
    parser.add_argument('--config', default='dressing_change_demo/configs/tsm_config.py', help='train config file path')
    parser.add_argument('--checkpoint', default='tsm_imagenet-pretrained-r50_8xb16-1x1x16-50e_sthv2-rgb_20230317-ec6696ad.pth', help='checkpoint file')
    parser.add_argument('--data_root', default='extracted_data3_p', help='root directory for data')
    parser.add_argument('--work_dir', default='work_dirs/demo', help='the dir to save logs and models')
    parser.add_argument('--action_class', default='6', help='action class to train on')
    parser.add_argument('--epochs', type=int, default=200, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--device', default='cuda:1', help='device used for training')
    return parser.parse_args()

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(os.path.dirname(filename), 'model_best.pth.tar'))

def main():
    args = parse_args()
    
    # Register mmaction modules
    register_all_modules(init_default_scope=True)
    
    # Setup logging
    if not os.path.exists(args.work_dir):
        os.makedirs(args.work_dir)
    
    logging.basicConfig(filename=os.path.join(args.work_dir, 'training.log'), level=logging.INFO)
    writer = SummaryWriter(args.work_dir)
    
    # Build Model
    print(f"Building model from {args.config}...")
    model = init_recognizer(args.config, args.checkpoint, device=args.device)
    
    # Freeze layers
    for param in model.parameters():
        param.requires_grad = False
    for param in model.backbone.layer4.parameters():
        param.requires_grad = True
        
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=1e-4)
    
    # Data Loaders
    # We need to construct the config for dataloaders manually as in the original script
    # or load from config file if we put it there.
    # The original script defines them inline. We'll do the same here for clarity/demo.
    
    file_client_args = dict(io_backend='disk')
    sthv2_flip_label_map = {86: 87, 87: 86, 93: 94, 94: 93, 166: 167, 167: 166}

    train_pipeline_cfg = [
        dict(type='LoadVideoDir', **file_client_args),
        dict(type='SampleFrames', clip_len=1, frame_interval=1, num_clips=16),
        dict(type='RawFrameDecode', io_backend='disk', decoding_backend='cv2'),
        dict(type='Resize', scale=(-1, 256)),
        dict(
            type='MultiScaleCrop',
            input_size=224,
            scales=(1, 0.875, 0.75, 0.66),
            random_crop=False,
            max_wh_scale_gap=1,
            num_fixed_crops=13),
        dict(type='Resize', scale=(224, 224), keep_ratio=False),
        dict(type='Flip', flip_ratio=0.5, flip_label_map=sthv2_flip_label_map),
        dict(type='FormatShape', input_format='NCHW'),
        dict(type='PackActionInputs')
    ]

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
    
    train_dataloader_cfg = dict(
        batch_size=args.batch_size,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        sampler=dict(type='DefaultSampler', shuffle=True),
        dataset=dict(
            type=dataset_type,
            data_root=args.data_root,
            action_path=args.action_class,
            pre='train',
            pipeline=train_pipeline_cfg,
            ))   

    support_dataloader_cfg = dict(
        batch_size=args.batch_size,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        sampler=dict(type='DefaultSampler', shuffle=True),
        dataset=dict(
            type=dataset_type,
            data_root=args.data_root,
            action_path=args.action_class,
            pre='support',
            pipeline=val_pipeline_cfg,
            test_mode=True))

    print("Building dataloaders...")
    train_data_loader = Runner.build_dataloader(dataloader=train_dataloader_cfg)
    support_data_loader = Runner.build_dataloader(dataloader=support_dataloader_cfg)
    
    # Training Loop
    print("Starting training...")
    model.train()
    best_acc = 0.0
    avg_pool = nn.AdaptiveAvgPool2d((1, 1))
    
    temperature = 0.07
    base_temperature = 0.07
    ratio = 0.5
    
    for epoch in range(args.epochs):
        try:
            sup_datch = next(iter(support_data_loader))
        except StopIteration:
            # Restart support loader if exhausted (though it should be infinite with persistent_workers?)
            # DefaultSampler with shuffle=True usually is infinite if configured right or we just re-iter.
            # But here we just take one batch per epoch? No, inside the loop.
            # The original code does `sup_datch = next(iter(support_data_loader))` ONCE per epoch.
            # This means it uses the SAME support batch for the whole epoch?
            # "sup_datch = next(iter(support_data_loader))" is inside "for epoch_counter in range(args.epochs):"
            # So yes, one support batch per epoch.
            pass
            
        sup_labels = [item.gt_label for item in sup_datch['data_samples']]
        sup_labels = torch.stack(sup_labels).to(args.device)
        sup_data = model.data_preprocessor(sup_datch, training=True)

        epoch_train_support_acc = 0.0
        total_batches = 0
        total_loss_epoch = 0.0
        
        for data_batch in train_data_loader:
            train_batch = data_batch
            data = {}
            
            train_data = model.data_preprocessor(train_batch, training=True)
            train_labels = torch.stack([item.gt_label for item in train_batch['data_samples']]).to(args.device)

            data['inputs'] = torch.cat([train_data['inputs'], sup_data['inputs']], dim=0)

            features = model(**data, mode='tensor')
            len_train = train_data['inputs'].shape[0] * train_data['inputs'].shape[1]
            train_features = features[:len_train]
            support_features = features[len_train:]

            train_features = avg_pool(train_features)
            train_features = train_features.view(train_labels.shape[0], -1)
            train_features = F.normalize(train_features, dim=1)

            support_features = avg_pool(support_features)
            support_features = support_features.view(sup_labels.shape[0], -1)
            support_features = F.normalize(support_features, dim=1)

            loss_train_train = intra_contrastive_loss(train_features, train_labels, temperature, base_temperature)
            loss_support_support = intra_contrastive_loss(support_features, sup_labels, temperature, base_temperature)
            loss_train_support, preds = cross_contrastive_loss(train_features, train_labels, support_features, sup_labels, temperature, base_temperature)

            cross_acc = (sup_labels[preds.squeeze()] == train_labels).float().mean()
            
            epoch_train_support_acc += cross_acc.item()
            total_batches += 1

            total_loss = ratio * loss_train_support + (1-ratio) * (0.5 * loss_train_train +  0.5 * loss_support_support)
            total_loss_epoch += total_loss.item()

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

        avg_acc = epoch_train_support_acc / total_batches if total_batches > 0 else 0
        avg_loss = total_loss_epoch / total_batches if total_batches > 0 else 0
        
        print(f"Epoch {epoch}, Loss: {avg_loss:.4f}, Top1 Acc: {avg_acc:.4f}")
        logging.info(f"Epoch {epoch}, Loss: {avg_loss}, Top1 Acc: {avg_acc}")
        
        writer.add_scalar('loss', avg_loss, global_step=epoch)
        writer.add_scalar('acc/top1', avg_acc, global_step=epoch)
        
        if avg_acc > best_acc:
            best_acc = avg_acc
            save_checkpoint({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, is_best=True, filename=os.path.join(args.work_dir, 'checkpoint_best_acc.pth.tar'))

    print("Training completed.")

if __name__ == '__main__':
    main()
