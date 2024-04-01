import os
import time
import cv2 as cv
import numpy as np
from tqdm.auto import tqdm
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from utils import MediaPipeTracker, GeneralUtils
from constants import (TRAINING_LOSSES, VALIDATION_LOSSES, NO_IMPROVEMENT_COUNT, BEST_EPOCH)


class ActionDetector:
    def __init__(self, class_mapping: dict, frames_interval: int = 10, model_path: str = None,
                 include_pose: bool = True, pose_positions: list = None):
        """ """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.class_mapping = class_mapping
        self.frames_interval = frames_interval
        self.include_pose = include_pose
        self.best_val_loss = float('inf')
        self.date_now_srt = datetime.now().strftime("%m%d%Y")
        self.pose_positions = pose_positions if pose_positions else [15, 13, 11, 0, 12, 14, 16]
        self.epochs, self.batch_size, self.learning_rate, self.early_stopping_patience = 0, 0, 0, 0

        self.input_size = (21 * 6) + (len(self.pose_positions) * 4)
        self.n_classes = len(self.class_mapping)

        self.model = CollabModel(input_size=self.input_size, n_classes=self.n_classes, device=self.device,
                                 include_pose=self.include_pose, pose_positions=self.pose_positions)
        self.mp_tracker = self.model.mp_model.mp_holistic.mp_tracker
        if model_path:
            self.load_model(model_path=model_path)

    def train(self, train_data: any, epochs: int, batch_size: int, learning_rate: float = 1e-3,
              validation_data: any = None, model_save_path: str = 'models', early_stopping_patience: int = 25,
              test: bool = True, test_data: any = None):
        # TODO: optimize
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.early_stopping_patience = early_stopping_patience

        losses = {TRAINING_LOSSES: [], VALIDATION_LOSSES: []}
        best_model_stats = {NO_IMPROVEMENT_COUNT: 0, BEST_EPOCH: 0}

        full_model_save_path = GeneralUtils.get_unique_filename(
            os.path.join(model_save_path,
                         f'model_{self.date_now_srt}_{epochs}e_{len(train_data)}d')
        )
        print(f'Model will be saved to {full_model_save_path}.')

        data_loader = DataLoader(dataset=train_data, batch_size=self.batch_size, shuffle=False)
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        self.model.train()
        start_time = time.time()
        for epoch in range(epochs):
            train_results = self._train_epoch(
                data_loader=data_loader, optimizer=optimizer, epoch=epoch, losses=losses, start_time=start_time,
                full_model_save_path=full_model_save_path, best_model_stats=best_model_stats,
                validation_data=validation_data)
            if train_results:
                optimizer, losses, best_model_stats = train_results
            else:
                break  # Early Stop

        GeneralUtils.compare_line_plot(data=losses[TRAINING_LOSSES],
                                       data_label='Training loss',
                                       compare_data=losses[VALIDATION_LOSSES],
                                       compare_data_label='Validation loss')
        if test:
            if test_data:
                self.test(test_data=test_data)
            else:
                print('Bypassing test since there is no test_data.')
        print(f'Training Done. '
              f'Best weights: Validation loss: {self.best_val_loss} on epoch: {best_model_stats[BEST_EPOCH]}.')
        return losses[TRAINING_LOSSES], losses[VALIDATION_LOSSES]

    def test(self, test_data):
        self.model.eval()
        with torch.no_grad():
            data_loader = DataLoader(dataset=test_data, batch_size=self.batch_size, shuffle=False)
            correct, total = 0, 0
            with tqdm(data_loader, desc=f"Testing model.", unit="batch") as tbar:
                for batch in tbar:
                    kp_sequences, labels = batch
                    labels = labels.to(self.device)
                    predicted = self.model(kp_sequences=kp_sequences)
                    _, predicted_idx = torch.max(predicted, 1)
                    correct += (predicted_idx == labels).sum().item()
                    total += labels.size(0)
            accuracy = correct / total
            print(f'Model accuracy is: {correct} / {total} = {round(accuracy, 3) * 100} %')
            return accuracy

    def run(self, threshold: float = 0.75, is_live: bool = True, video_path: str = ''):
        if not video_path and not is_live:
            print('ERROR: video_path parameter must have a value if is_live=False.')
            return None

        self.model.eval()
        cap = cv.VideoCapture(0) if is_live else cv.VideoCapture(video_path)
        kp_sequences, prediction_history = [], []
        previous_action = None
        frame_no = 0
        while cap.isOpened():
            is_capture, frame = cap.read()
            if not is_capture:
                break

            # get keypoints
            results = self.mp_tracker.Detect(frame)
            key_points = self.mp_tracker.get_key_points(results=results, pose=self.include_pose,
                                                        pose_positions=self.pose_positions)
            kp_sequences.append(key_points)
            kp_sequences = kp_sequences[1:] if len(kp_sequences) > self.frames_interval else kp_sequences

            frame_no += 1
            if frame_no >= self.frames_interval:
                frame_no = 0
                predicted = self.model(kp_sequences=torch.tensor(np.array([kp_sequences])))
                action = list(self.class_mapping.keys())[torch.argmax(predicted)]

                if action != previous_action:
                    previous_action = action
                    print(f'Action is: {action} | {predicted}')

            cv.imshow('Capture Action', frame)
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv.destroyAllWindows()

    def load_model(self, model_path):
        self.model.load_state_dict(torch.load(f=model_path))
        print(f'Model loaded from {model_path}')

    def _train_epoch(self, data_loader, optimizer, epoch, losses, full_model_save_path, best_model_stats, start_time,
                     validation_data):
        running_loss = 0.0
        with tqdm(data_loader, desc=f"Epoch {epoch + 1}/{self.epochs}", unit="batch") as tbar:
            for batch in tbar:
                running_loss, optimizer = self.__training_loop(batch=batch, optimizer=optimizer,
                                                               running_loss=running_loss)
            average_loss = running_loss / len(data_loader)
            losses[TRAINING_LOSSES].append(average_loss)

        # Validation
        val_loss, losses = self._validate(validation_data=validation_data, average_loss=average_loss, losses=losses,
                                          epoch=epoch)

        # Save the model if it has the best validation loss so far
        best_model_stats = self._save_best_model(val_loss=val_loss, epoch=epoch,
                                                 full_model_save_path=full_model_save_path,
                                                 best_model_stats=best_model_stats)
        self._save_training_data(epoch=epoch, training_loss=average_loss, val_loss=val_loss, start_time=start_time,
                                 full_model_save_path=full_model_save_path)

        if self._is_early_stop(no_improvement_count=best_model_stats[NO_IMPROVEMENT_COUNT], epoch=epoch):
            return None

        return optimizer, losses, best_model_stats

    def _is_early_stop(self, no_improvement_count, epoch):
        if 0 < self.early_stopping_patience <= no_improvement_count:
            print(f'No improvements after {self.early_stopping_patience} epochs. '
                  f'Training will stop at epoch {epoch}/{self.epochs}.')
            return True
        return False

    def _validate(self, validation_data, average_loss, losses, epoch):
        self.model.eval()
        data_loader = DataLoader(dataset=validation_data, batch_size=self.batch_size, shuffle=True)
        val_loss = 0
        with torch.no_grad():
            with tqdm(data_loader, desc=f"Validating:", unit="batch") as tbar:
                for batch in tbar:
                    val_loss = self.__validation_loop(batch, val_loss)
            avg_val_loss = val_loss / len(data_loader)
        self.model.train()
        print(f"Epoch {epoch + 1}/{self.epochs}, Loss: {average_loss}, Val Loss: {avg_val_loss}")
        losses['validation_losses'].append(avg_val_loss)
        return avg_val_loss, losses

    def _save_best_model(self, val_loss, epoch, full_model_save_path, best_model_stats):
        if val_loss < self.best_val_loss and epoch > 1:
            self.best_val_loss = val_loss
            best_model = self.model.state_dict()
            best_model_stats['best_epoch'] = epoch
            best_model_stats['no_improvement_count'] = 0
            best_model_full_save_path = os.path.join(full_model_save_path, 'best_model.pt')
            print(f'Best weights: Validation loss: {self.best_val_loss}')
            print(f'Best model will be saved to {best_model_full_save_path}.')
            torch.save(obj=best_model, f=best_model_full_save_path)
        else:
            best_model_stats['no_improvement_count'] += 1
        return best_model_stats

    def _save_training_data(self, full_model_save_path, epoch, training_loss, val_loss, start_time, model_name='model'):
        if not os.path.exists(full_model_save_path):
            os.makedirs(full_model_save_path)
        torch.save(obj=self.model.state_dict(), f=os.path.join(full_model_save_path, f'{model_name}.pt'))

        log_data_path = os.path.join(full_model_save_path, 'train_logs.txt')
        log_data = self._get_training_results(training_loss=training_loss, val_loss=val_loss,
                                              log_data_path=log_data_path, epoch=epoch)

        if epoch + 1 >= self.epochs:
            end_time = time.time()
            total_time = round(end_time - start_time, 3)
            log_data.append(f'Total train Time: {total_time}s')
            print(f'Total Train time: {total_time}')
        with open(log_data_path, 'w') as file:
            for item in log_data:
                file.write(str(item) + '\n')

    def _get_training_results(self, training_loss, val_loss, log_data_path, epoch):
        log_data = []
        if os.path.exists(log_data_path):
            with open(log_data_path, 'r') as file:
                log_data = [line.strip() for line in file.readlines()]
        else:
            log_data.append(f'Input size: {self.input_size}')
            log_data.append(f'Batch Size: {self.batch_size}')
            log_data.append(f'Epochs: {self.epochs}')
            log_data.append(f'Learning rate: {self.learning_rate}')
            log_data.append(f'Frame count: {self.frames_interval}')
            log_data.append(f'Include Pose: {str(self.include_pose)}')
            log_data.append(f'Pose Positions: {" ".join(map(str, self.pose_positions))}')
            log_data.append('Training Logs: ------------------------------')
        log_data.append(f'e={epoch} ;training_loss={training_loss} ;validation_loss={val_loss}')
        return log_data

    def __training_loop(self, batch, optimizer, running_loss):
        kp_sequences, labels = batch
        mp_predicted = self.model(kp_sequences=kp_sequences)

        # Backpropagation
        optimizer.zero_grad()
        loss = F.cross_entropy(mp_predicted, labels.to(self.device).long())
        running_loss += loss.item()
        loss.backward()
        optimizer.step()
        return running_loss, optimizer

    def __validation_loop(self, batch, val_loss):
        kp_sequences, labels = batch
        mp_predicted = self.model(kp_sequences=kp_sequences)
        loss = F.cross_entropy(mp_predicted, labels.to(self.device).long())
        val_loss += loss.item()
        return val_loss


class CollabModel(nn.Module):
    def __init__(self, input_size: int, n_classes: int, device: torch.device = torch.device('cpu'),
                 include_pose: bool = False, pose_positions: list = None):
        super().__init__()
        self.device = device
        self.input_size = input_size
        self.n_classes = n_classes
        self.mp_model = MediaPipeModel(input_size=self.input_size, n_classes=self.n_classes, device=self.device,
                                       include_pose=include_pose, pose_positions=pose_positions)

    def forward(self, kp_sequences: list):
        predicted_lstm, predicted_cnn = self.mp_model(kp_sequences=kp_sequences)
        # TODO:
        return predicted_lstm
        """
        if predicted_lstm is not None and predicted_cnn is not None:
            # Get best prediction
            # TODO :
            predicted = predicted_lstm
            return predicted
        return None"""


class MediaPipeModel(nn.Module):
    def __init__(self, input_size: int, n_classes: int, device: torch.device = torch.device('cpu'),
                 include_pose: bool = False, pose_positions: list = None):
        super().__init__()
        self.device = device
        self.input_size = input_size
        self.n_classes = n_classes
        self.include_pose = include_pose
        self.pose_positions = pose_positions
        self.mp_holistic = MediaPipeHolisticUnit()
        self.sequence_classifier_lstm = SequenceClassifierLSTM(input_size=self.input_size, n_classes=self.n_classes,
                                                               device=self.device)
        self.sequence_classifier_cnn = SequenceClassifierCNN(input_size=self.input_size, n_classes=self.n_classes,
                                                             device=self.device)
        self.sequence_classifier_lstm = self.sequence_classifier_lstm.to(self.device)
        self.sequence_classifier_cnn = self.sequence_classifier_cnn.to(self.device)

    def forward(self, kp_sequences: list):
        predicted_lstm = self.sequence_classifier_lstm(kp_sequences)
        #predicted_cnn = self.sequence_classifier_cnn(kp_sequences)
        predicted_cnn = None
        return predicted_lstm, predicted_cnn


class MediaPipeHolisticUnit:
    def __init__(self):
        """Using MediaPipe's Holistic Unit model to gather specific key points from Face, Hands, and Pose"""
        self.mp_tracker = MediaPipeTracker()

    def process(self, frame: any, pose: bool = False, pose_positions: list = None) -> list:
        results = self.mp_tracker.Detect(frame)
        key_points = self.mp_tracker.get_key_points(results=results, pose=pose, pose_positions=pose_positions)
        return key_points


class SequenceClassifierLSTM(nn.Module):
    def __init__(self, input_size: int, n_classes: int, batch_first: bool = True,
                 device: torch.device = torch.device('cpu')):
        super().__init__()
        self.device = device
        self.input_size = input_size
        self.initial_hidden_layer = 16

        self.lstm1 = nn.LSTM(input_size, self.initial_hidden_layer, batch_first=batch_first)
        self.fc = nn.Sequential(
            nn.Linear(self.initial_hidden_layer, 32),
            nn.ReLU(),
            nn.Linear(32, n_classes)
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, kp_sequence: torch.Tensor):
        kp_sequence = kp_sequence.to(self.device).float()
        out, _ = self.lstm1(kp_sequence)
        out = self.fc(out[:, -1, :])  # Linear Layers
        return self.softmax(out)


# TODO
class SequenceClassifierCNN(nn.Module):
    def __init__(self, input_size: int, n_classes: int, batch_first: bool = True,
                 device: torch.device = torch.device('cpu')):
        super().__init__()
        self.device = device
        self.input_size = input_size
        self.lstm1 = nn.LSTM(self.input_size, 64, batch_first=batch_first)
        self.lstm2 = nn.LSTM(64, 128, batch_first=batch_first)
        self.lstm3 = nn.LSTM(128, 64, batch_first=batch_first)
        self.fc1 = nn.Linear(64, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, n_classes)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, kp_sequence: torch.Tensor):
        kp_sequence = kp_sequence.to(self.device).float()
        out, _ = self.lstm3(self.relu(self.lstm2(self.relu(self.lstm1(kp_sequence)[0]))[0]))  # LSTM layers
        out = self.fc3(self.relu(self.fc2(self.relu(self.fc1(self.relu(out[:, -1, :]))))))  # Linear Layers
        return self.softmax(out)
