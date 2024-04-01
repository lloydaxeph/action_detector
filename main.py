import os
from torch.utils.data import random_split

from utils import CaptureUtils
from models import ActionDetector
from custom_dataset import CustomDataset

if __name__ == '__main__':
    PROJECT_PATH = r''

    action = 'rock_n_roll'
    frame_count = 15
    data_path = os.path.join(PROJECT_PATH, 'Data', 'ASL_test_data')

    num_sample = 20
    batch_size = 24
    epochs = 500
    model_path = os.path.join(PROJECT_PATH, 'Projects', 'Applications', 'Sign_Language_Project', 'v1', 'models',
                              'model_03312024_500e_54d')
    cap_utils = CaptureUtils()
    predefined_ds = os.path.join(data_path, 'train2', 'ds_60d_03312024.pkl')
    #predefined_ds = None
    save_ds = False
    dataset = CustomDataset(data_path=os.path.join(data_path, 'train2'),
                            save_dataset=save_ds, preprocessed_dataset_path=predefined_ds)
    train_size = int(0.75 * len(dataset))
    val_test_size = len(dataset) - train_size

    train_ds, val_test = random_split(dataset, [train_size, val_test_size])
    val_ds, test_ds = random_split(val_test, [val_test_size//2, len(val_test) - val_test_size//2])
    mode = 'train'
    if mode == 'capture':
        for n in range(num_sample):
            cap_utils.capture_action(action=action, frame_count=frame_count, save_path=os.path.join(data_path, 'train2')
                                     ,samp_num=str(n + 1))
    elif mode == 'train':
        detector = ActionDetector(class_mapping=dataset.class_mapping, frames_interval=dataset.frames_interval,
                                  include_pose=dataset.include_pose, pose_positions=dataset.pose_positions)
        detector.train(train_data=train_ds, validation_data=val_ds, epochs=epochs, batch_size=batch_size,
                       early_stopping_patience=epochs, test=True, test_data=test_ds)
    elif mode == 'test':
        detector = ActionDetector(class_mapping=dataset.class_mapping, frames_interval=dataset.frames_interval,
                                  include_pose=dataset.include_pose, pose_positions=dataset.pose_positions,
                                  model_path=os.path.join(model_path, 'best_model.pt'))

        detector.run()


