import os
from torch.utils.data import random_split

from utils import CaptureUtils
from models import ActionDetector
from custom_dataset import CustomDataset
from constants import CNN, LSTM


def capture_data(data_save_path: str, num_sample: int, action: str, frame_count: int):
    """DATA GATHERING EXAMPLE"""
    cap = CaptureUtils()
    for n in range(num_sample):
        cap.capture_action(action=action, frame_count=frame_count, save_path=data_save_path, samp_num=str(n + 1))


def define_custom_dataset(data_save_path: str, train_size: int, is_save_data: bool = True,
                          preprocessed_dataset_path: str = None) \
        -> (CustomDataset, CustomDataset, CustomDataset, CustomDataset):
    """EXAMPLE CUSTOM DATASET IMPLEMENTATION"""
    dataset = CustomDataset(data_path=data_save_path, save_dataset=is_save_data,
                            preprocessed_dataset_path=preprocessed_dataset_path)
    val_test_size = len(dataset) - train_size
    train_ds, val_test = random_split(dataset, [train_size, val_test_size])
    val_ds, test_ds = random_split(val_test, [val_test_size // 2, len(val_test) - val_test_size // 2])
    return dataset, train_ds, val_ds, test_ds


def train_model(data_save_path: str, train_size: int, model_type: str, initial_hidden_layer: int, epochs: int,
                batch_size: int, early_stopping_patience: int, is_save_data: bool = True,
                preprocessed_dataset_path: str = None, is_test: bool = True):
    """MODEL TRAIN EXAMPLE"""
    dataset, train_ds, val_ds, test_ds = define_custom_dataset(data_save_path=data_save_path, train_size=train_size,
                                                               is_save_data=is_save_data,
                                                               preprocessed_dataset_path=preprocessed_dataset_path)
    detector = ActionDetector(class_mapping=dataset.class_mapping, frames_interval=dataset.frames_interval,
                              include_pose=dataset.include_pose, pose_positions=dataset.pose_positions,
                              model_type=model_type, initial_hidden_layer=initial_hidden_layer)
    detector.train(train_data=train_ds, validation_data=val_ds, epochs=epochs, batch_size=batch_size,
                       early_stopping_patience=early_stopping_patience, test=is_test, test_data=test_ds)


def demo(model_path: str, data_save_path: str, train_size: int, is_save_data: bool = True,
         preprocessed_dataset_path: str = None):
    """MODEL DEMONSTRATION"""
    dataset, train_ds, val_ds, test_ds = define_custom_dataset(data_save_path=data_save_path, train_size=train_size,
                                                               is_save_data=is_save_data,
                                                               preprocessed_dataset_path=preprocessed_dataset_path)
    detector = ActionDetector(class_mapping=dataset.class_mapping, frames_interval=dataset.frames_interval,
                              include_pose=dataset.include_pose, pose_positions=dataset.pose_positions,
                              model_path=model_path)
    detector.run()


if __name__ == '__main__':
    PROJECT_PATH = '.../project'
    data_save_path = os.path.join(PROJECT_PATH, 'train_data')

    # DATA GATHERING EXAMPLE -------------------------------------------------------------------------------------------
    action = 'hello world'
    frame_count = 15
    num_sample = 15
    capture_data(data_save_path=data_save_path, num_sample=num_sample, action=action, frame_count=frame_count)
    # ------------------------------------------------------------------------------------------------------------------

    # MODEL TRAIN EXAMPLE ----------------------------------------------------------------------------------------------
    train_size = 100 # depends on how large your dataset is
    batch_size = 24
    epochs = 100
    initial_hidden_layer = 16
    model_type = CNN
    early_stopping_patience = epochs
    preprocessed_dataset_path = os.path.join(data_save_path, 'my_predefined_ds.pkl')
    is_save_data = True

    train_model(data_save_path=data_save_path, train_size=train_size, model_type=model_type,
                initial_hidden_layer=initial_hidden_layer, epochs=epochs, batch_size=batch_size,
                early_stopping_patience=early_stopping_patience, is_save_data=is_save_data,
                preprocessed_dataset_path=preprocessed_dataset_path)
    # ------------------------------------------------------------------------------------------------------------------

    # MODEL DEMONSTRATION ----------------------------------------------------------------------------------------------
    model_path = os.path.join(PROJECT_PATH, 'my_model.pt')
    demo(model_path=model_path, data_save_path=data_save_path, train_size=train_size, is_save_data=is_save_data,
         preprocessed_dataset_path=preprocessed_dataset_path)
    # ------------------------------------------------------------------------------------------------------------------
