import os
import os.path as osp
import pandas as pd
import numpy as np
from typing import Optional, Dict
from mmaction.registry import DATASETS, TRANSFORMS
from mmaction.datasets import VideoDataset
from mmcv.transforms import BaseTransform

def find_mp4_files(root_folder):
    mp4_files = []
    for root, dirs, files in os.walk(root_folder):
        for file in files:
            if file.endswith('.mp4'):
                mp4_files.append(os.path.join(root, file))
    return mp4_files

def read_video_info(root_path, action_path):
    summary = "summary.xlsx"
    summary_file_path = os.path.join(root_path, summary)
    video_info = []
    
    if "3.1" in action_path:
        label_index = 8
    elif "3.2" in action_path:
        label_index = 9
    elif "5.2" in action_path:
        label_index = 12
    elif "6" in action_path:
        label_index = 13
    else:
        label_index = 15

    if not os.path.exists(summary_file_path):
        print(f"Warning: {summary_file_path} not found.")
        return []

    df = pd.read_excel(summary_file_path)

    item_path = os.path.join(root_path, action_path)
    if not os.path.exists(item_path):
        print(f"Warning: {item_path} not found.")
        return []
        
    mp4_files = [f for f in os.listdir(item_path)]
    

    for index, row in df.iterrows():
        if not pd.isna(row.iloc[1]):
            file_name = row.iloc[1]
        else:
            file_name = row.iloc[0]
        if len(file_name.split('-'))>3:
            file_name = '-'.join(file_name.split('-')[:3])
            
        label = row.iloc[label_index]
        ff = 0
        for mp4_file in mp4_files:
            if file_name in mp4_file:
                ff = 1
                if "-sj" in mp4_file:
                    ff = 2
                    video_info.append({
                        "file": mp4_file,
                        "label": label
                    })
                    break
        if ff == 0:
            # print(f"File {file_name} not found")
            pass
        elif ff == 1:
            # print(f"File {file_name} found, but not a '-sj' file.")
            pass

    sorted_video_info = sorted(video_info, key=lambda x: x["file"])
    return sorted_video_info

@DATASETS.register_module()
class DatasetZelda(VideoDataset):
    def __init__(self, pipeline, data_root, action_path, pre,
                 multi_class: bool = False,
                 num_classes: Optional[int] = None,
                 start_index: int = 0,
                 test_mode: bool = False,
                 delimiter: str = ' ',
                 ann_file = '',
                 modality='RGB', **kwargs)->None:
        self.modality = modality
        self.pre = pre
        self.data_root = data_root
        self.data_list = self.get_data_list(self.data_root, action_path)
        self.train_data, self.support_set, self.test_data = self.divide_data_list(self.data_list, action_path)
        super(DatasetZelda, self).__init__(pipeline=pipeline,data_root=data_root,multi_class=multi_class,num_classes=num_classes,modality=modality,
            start_index=start_index,ann_file=ann_file, test_mode=test_mode, **kwargs)

    def get_data_list(self,data_root, action_path='6'):
        
        data_list = []
        video_info = read_video_info(data_root, action_path)
        i = 0
        new_video_info_sort = sorted(video_info, key=lambda x: x["file"])

        while i < len(new_video_info_sort):
            
            label = new_video_info_sort[i]['label']
            file_name = new_video_info_sort[i]['file']
            if '-sj' in file_name:
                video_path = osp.join(data_root, action_path, file_name)
                filename = video_path
                data_list.append(dict(filename=filename, label=int(label)))
            i = i + 1
        return data_list
    
    def divide_data_list(self, data_list, action_path='6'):
        if "3.1" in action_path:
            data_split_path = "dataset_splits_3.1"
        elif "3.2" in action_path:
            data_split_path = "dataset_splits_3.2"
        elif "5.2" in action_path:
            data_split_path = "dataset_splits_5.2"
        elif "6" in action_path:
            data_split_path = "dataset_splits_6"
        else:
            data_split_path = "dataset_splits_overall"


        support_file = osp.join(self.data_root, data_split_path, 'support_set.csv')
        train_file = osp.join(self.data_root, data_split_path, 'train_set.csv')
        test_file = osp.join(self.data_root, data_split_path, 'test_set.csv')

        support_set = []
        train_data = []
        test_data = []

        
        if os.path.exists(support_file):

            support_list = pd.read_csv(support_file, header=None).values.tolist()
            train_list = pd.read_csv(train_file, header=None).values.tolist()
            test_list = pd.read_csv(test_file, header=None).values.tolist()

            support_list_set = []
            train_list_set = []
            test_list_set = []

            for item in support_list:
                if not pd.isna(item[1]):
                    support_list_set.append(item[1])
                else:
                    support_list_set.append(item[0])
            for item in train_list:
                if not pd.isna(item[1]):
                    train_list_set.append(item[1])
                else:
                    train_list_set.append(item[0])
            for item in test_list:
                if not pd.isna(item[1]):
                    test_list_set.append(item[1])
                else:
                    test_list_set.append(item[0])

            for i in range(len(data_list)):
                ff = 0
                for j in range(len(support_list_set)):
                    if support_list_set[j] in data_list[i]['filename']:
                        ff = 1
                        support_set.append(data_list[i])
                        break
                if ff == 0:
                    for k in range(len(train_list_set)):
                        if train_list_set[k] in data_list[i]['filename']:
                            ff = 1
                            train_data.append(data_list[i])
                            break
                if ff == 0:
                    for l in range(len(test_list_set)):
                        if test_list_set[l] in data_list[i]['filename']:
                            ff = 1
                            test_data.append(data_list[i])
                            break
                if ff == 0:
                    # print(f"File {data_list[i]['filename']} not found in support, train or test set.")       
                    pass

        return train_data, support_set, test_data
        
        
    def load_data_list(self):
        
        if self.pre == 'train':
            data = self.train_data
            print(f"Size of train set: {len(data)}")
        elif self.pre == 'support':        
            data = self.support_set
            print(f"Size of support set: {len(data)}")
        else:
            data = self.test_data
            print(f"Size of test set: {len(data)}")
        return data


    def get_data_info(self, idx: int) -> dict:
        data_info = super().get_data_info(idx)
        data_info['modality'] = self.modality
        return data_info

@TRANSFORMS.register_module()
class LoadVideoDir(BaseTransform):
    def __init__(self,
                 filename_tmpl: str = 'frame_{:04d}.jpg',
                 start_index: int = 0,
                 io_backend: str = 'disk',
                 modality: str = 'RGB',
                 **kwargs) -> None:
        self.filename_tmpl = filename_tmpl
        self.start_index = start_index
        self.io_backend = io_backend
        self.modality = modality
        self.kwargs = kwargs
        self.file_client = None
        
    def _count_frames(self, frame_dir: str) -> int:
        """Count total frames in the directory."""
        try:

            return len(os.listdir(frame_dir))
        except OSError:
            # print(f"Error accessing directory: {frame_dir}.")
            return 0
    
    def transform(self, results: Dict) -> Dict:
        """Initialize folder reading."""

        frame_dir = results['filename']
        total_frames = self._count_frames(frame_dir)
        
        results['frame_dir'] = frame_dir
        results['total_frames'] = total_frames
        results['start_index'] = self.start_index
        results['filename_tmpl'] = self.filename_tmpl
        results['modality'] = self.modality
        results['avg_fps'] = 5.0

        return results
