import os
import numpy as np
from tqdm.auto import tqdm
from datetime import datetime
from torch.utils.data import Dataset

from utils import CaptureUtils, MediaPipeTracker, GeneralUtils
from constants import (KEY_POINTS, LABELS)


class CustomDataset(Dataset):
    def __init__(self, data_path: str, custom_cmap_path: str = None, frames_interval: int = None,
                 include_pose: bool = True, pose_positions: list = None, skip_preprocess: bool = False,
                 preprocessed_dataset_path: str = None, save_dataset: bool = False):
        """
        Input are video files and outputs their respective frames.
        Valid extensions: [mp4, avi, mkv, mov, wmv]
        """
        self.data_path = data_path
        self.mp_tracker = MediaPipeTracker()
        self.include_pose = include_pose
        self.pose_positions = pose_positions if pose_positions else [15, 13, 11, 0, 12, 14, 16]
        self.skip_preprocess = skip_preprocess
        self.save_dataset = save_dataset
        self.preprocessed_dataset_path = preprocessed_dataset_path

        self.video_extensions = ['.mp4', '.avi', '.mkv', '.mov', '.wmv']
        self.class_mapping_path = custom_cmap_path if custom_cmap_path else os.path.join(data_path, 'class_mapping.txt')
        self.class_mapping = self._get_class_mappings(class_mapping_path=self.class_mapping_path)

        self.data = self._get_data()
        self.frames_interval = frames_interval if frames_interval else self._get_default_frame_interval()

    def __len__(self) -> int:
        return len(self.data[LABELS])

    def __getitem__(self, idx: int) -> (list, list):
        # TODO : Have an option to perform data augmentation
        return self.data[KEY_POINTS][idx], self.data[LABELS][idx]

    def _get_data(self) -> dict:
        """Reads every frame in a video on the data_path directory."""
        if self.preprocessed_dataset_path:
            full_data = GeneralUtils.load_data_from_pickle(file_path=self.preprocessed_dataset_path)
            return full_data

        video_file_list = GeneralUtils.verify_file_list(file_list=os.listdir(self.data_path),
                                                        valid_extensions=self.video_extensions)
        video_file_list = [video_file_list[0]] if self.skip_preprocess else video_file_list

        if not video_file_list:
            raise FileNotFoundError(f'No valid video files exists in {self.data_path}')
        full_data = {KEY_POINTS: [], LABELS: []}
        with tqdm(video_file_list, desc=f'Preprocessing data.', unit="video_file") as tbar:
            for video_file in tbar:
                frames_sequence = self._get_frame_data(video_file=video_file)
                key_points_sequences = self._process_frames(frames_sequence=frames_sequence)
                label_idx = self._get_label_idx(label=video_file.split('-')[0])

                full_data[KEY_POINTS].append(key_points_sequences)
                full_data[LABELS].append(label_idx)
        print('Preprocess done.')
        if self.save_dataset:
            self._save_preprocess_dataset(data=full_data)
        return full_data

    def _save_preprocess_dataset(self, data: dict):
        date_now_srt = datetime.now().strftime("%m%d%Y")
        ds_file_name = GeneralUtils.get_unique_filename(file_path=f'ds_{len(data[LABELS])}d_{date_now_srt}.pkl')
        GeneralUtils.save_data_to_pickle(data=data, save_path=os.path.join(self.data_path, ds_file_name))

    def _process_frames(self, frames_sequence):
        key_points_sequences = []
        for frame in frames_sequence:
            results = self.mp_tracker.Detect(frame)
            key_points = self.mp_tracker.get_key_points(results=results, pose=self.include_pose,
                                                        pose_positions=self.pose_positions)
            key_points_sequences.append(key_points)
        return np.array(key_points_sequences)

    def _get_frame_data(self, video_file: str) -> list:
        abs_video_path = os.path.join(self.data_path, video_file)
        frames = CaptureUtils.read_frames(video_path=abs_video_path)
        return frames

    def _get_label_idx(self, label: str) -> int:
        if label not in list(self.class_mapping.keys()):
            raise Exception(f'{label} is not in class mapping.')
        return self.class_mapping[label]

    def _get_default_frame_interval(self):
        resize_data = False
        min_frames = len(self.data[KEY_POINTS][0])
        for item in self.data[KEY_POINTS][1:]:
            item_len = len(item)
            if item_len != min_frames:
                resize_data = True
                if item_len < min_frames:
                    min_frames = item_len

        if resize_data:
            self.data[KEY_POINTS] = self._resize_data(min_frames=min_frames)
        return min_frames

    def _resize_data(self, min_frames):
        print(f'Data not unified in size, will resize down to {min_frames} frames.')
        return [d[:min_frames] for d in self.data[KEY_POINTS]]

    def _get_class_mappings(self, class_mapping_path: str) -> dict:
        if os.path.isfile(class_mapping_path):
            with open(class_mapping_path) as f:
                class_mapping = {line.rstrip(): idx for idx, line in enumerate(f)}
                return class_mapping
        else:
            raise FileNotFoundError(f'{self.class_mapping_path} does not exits for class_mapping file.')
