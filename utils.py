import os
import pickle
import cv2 as cv
import numpy as np
import mediapipe as mp
from datetime import datetime
import matplotlib.pyplot as plt


from constants import (POSE, FACE, LEFT, RIGHT)


class GeneralUtils:
    @staticmethod
    def verify_file_list(file_list: list, valid_extensions) -> list:
        verified_list = []
        for file_name in file_list:
            file_extension = os.path.splitext(file_name)[1].lower()
            if file_extension in valid_extensions:
                verified_list.append(file_name)
        return verified_list

    @staticmethod
    def compare_line_plot(data, data_label, compare_data, compare_data_label):
        plt.plot(data, label=data_label)
        plt.plot(compare_data, label=compare_data_label)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

    @staticmethod
    def create_custom_dir(directory_path: str, use_unique: bool = False) -> str:
        """Creates a new directory (directory_path).
        If path name already exist, it adds a numerical suffix to the directory name"""
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
            return directory_path
        else:
            if use_unique:
                count = 1
                while True:
                    new_directory_path = f"{directory_path}_{count}"
                    if not os.path.exists(new_directory_path):
                        os.makedirs(new_directory_path)
                        return new_directory_path
                    count += 1
            else:
                return directory_path

    @staticmethod
    def get_unique_filename(file_path: str) -> str:
        """Checks if file_path already exists.
        If it does, it adds a numerical suffix to the file name"""
        if not os.path.exists(file_path):
            return file_path
        file_name, file_ext = os.path.splitext(file_path)
        count = 1
        while os.path.exists(f"{file_name}_{count}{file_ext}"):
            count += 1
        return f"{file_name}_{count}{file_ext}"

    @staticmethod
    def update_class_mapping(label: str, data_path: str, cmap_file_name='class_mapping'):
        cmap_file_path = os.path.join(data_path, f'{cmap_file_name}.txt')
        if not os.path.isfile(cmap_file_path):
            with open(cmap_file_path, 'w') as file:
                file.write(f"""{label}""")
            print(f'New class_mapping file is now created: {cmap_file_path}')
        else:
            with open(cmap_file_path, 'r+') as file:
                content = file.read().strip().split('\n')
                if label not in content:
                    file.write(f"\n{label}")
                    print(f'Updated "{label}" into {cmap_file_name}.txt.')
                else:
                    print(f'"{label}" already exists in {cmap_file_name}.txt.')

    @staticmethod
    def save_data_to_pickle(data: any, save_path: str):
        with open(save_path, 'wb') as file:
            pickle.dump(data, file)
            print(f'Data is saved to {save_path}')

    @staticmethod
    def load_data_from_pickle(file_path: str):
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
            print(f'Loaded data from {file_path}')
        return data


class MediaPipeTracker:
    def __init__(self, detect_conf: float = 0.5, tracking_conf: float = 0.5):
        """ """
        self.mp_drawing = mp.solutions.drawing_utils

        # Holistic Solution
        self.landmarks = {POSE: [], FACE: [], LEFT: [], RIGHT: []}
        self.mp_holistic = mp.solutions.holistic
        self.holistic_model = self.mp_holistic.Holistic(min_detection_confidence=detect_conf,
                                                        min_tracking_confidence=tracking_conf)

    def Detect(self, frame: np.ndarray, color: any = cv.COLOR_BGR2RGB):
        return self.holistic_model.process(cv.cvtColor(frame, color))

    def get_key_points(self, results: any, face: bool = False, pose: bool = True, pose_positions: list = None) \
            -> np.array:
        landmarks = self._get_positions(results=results)
        final_landmarks = []
        if pose:
            if pose_positions:
                landmarks[0] = [landmarks[0][p] for p in pose_positions]
            final_landmarks.append(landmarks[0])
        if face:
            final_landmarks.append(landmarks[1])
        final_landmarks.extend(landmarks[2:])
        return np.concatenate([np.array(part).flatten() for part in final_landmarks])

    def draw_landmarks(self, frame, results, pose: bool = True, left_hand: bool = True, right_hand: bool = True,
                       face: bool = True, color=((0, 0, 255), (255, 255, 255)), thickness=(1, 1), radius=(1, 1)):
        if len(color) == len(thickness) == len(radius):
            drawing_specs = [self.mp_drawing.DrawingSpec(color=color[i],
                                                         thickness=thickness[i],
                                                         circle_radius=radius[i]) for i in range(len(color))]
        else:
            raise AttributeError('color, thickness & radius parameter does not have the same length.')

        solutions = [pose, face, left_hand, right_hand]
        sol_results = [results.pose_landmarks, results.face_landmarks,
                       results.left_hand_landmarks, results.right_hand_landmarks]
        sol_connections = [self.mp_holistic.POSE_CONNECTIONS, self.mp_holistic.FACEMESH_TESSELATION,
                           self.mp_holistic.HAND_CONNECTIONS, self.mp_holistic.HAND_CONNECTIONS]
        for i, is_draw in enumerate(solutions):
            if is_draw:
                self._draw_landmarks(frame=frame, results=sol_results[i], connections=sol_connections[i],
                                     drawing_specs=drawing_specs)
        return frame

    def _draw_landmarks(self, frame, results, connections, drawing_specs):
        self.mp_drawing.draw_landmarks(frame, results, connections, drawing_specs[0], drawing_specs[1])

    def _get_positions(self, results: any) -> list:
        pose_landmarks = self._preprocess_landmarks(landmarks_results=results.pose_landmarks, default_zeros=(33, 4),
                                                    is_pose=True)
        face_landmarks = self._preprocess_landmarks(landmarks_results=results.face_landmarks, default_zeros=(468, 3))
        left_hand_landmarks = self._preprocess_landmarks(landmarks_results=results.left_hand_landmarks,
                                                         default_zeros=(21, 3))
        right_hand_landmarks = self._preprocess_landmarks(landmarks_results=results.right_hand_landmarks,
                                                          default_zeros=(21, 3))
        return [pose_landmarks, face_landmarks, left_hand_landmarks, right_hand_landmarks]

    def _preprocess_landmarks(self, landmarks_results: any, default_zeros: tuple, is_pose: bool = False) -> any:
        if landmarks_results:
            if is_pose:
                results = [[res.x, res.y, res.z, res.visibility] for res in landmarks_results.landmark]
            else:
                results = [[res.x, res.y, res.z] for res in landmarks_results.landmark]
        else:
            results = np.zeros(default_zeros)
        return results


class CaptureUtils:
    def __init__(self):
        self.mp_tracker = MediaPipeTracker()

    @staticmethod
    def read_frames(video_path: str) -> np.array:
        cap = cv.VideoCapture(video_path)
        if not cap.isOpened():  # Check if the video opened successfully
            raise ValueError(f'ERROR: Unable to open video file {video_path}.')

        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        return np.array(frames)

    def capture_action(self, action, frame_count: int, save_path, prep_duration: int = 3, codec: str = 'mp4v',
                       samp_num: str = ''):
        # TODO: verify action if have space
        time_stamp = datetime.now().strftime("%m%d%Y%H%M%S")
        save_path = GeneralUtils.create_custom_dir(directory_path=save_path)
        capture_file_path = GeneralUtils.get_unique_filename(
            file_path=os.path.join(save_path, f'{action}-{frame_count}f_{time_stamp}.mp4'))

        # Initial capture for preparation
        cap = cv.VideoCapture(0)
        cap = self._show_capture(cap=cap, duration=prep_duration,
                                 frame_text=f'Prepare to capture "{action}"', samp_num=samp_num)

        # Start Gather Data & Save as Video
        fourcc = cv.VideoWriter_fourcc(*codec)
        out = cv.VideoWriter(capture_file_path, fourcc, 20.0, (640, 480))
        GeneralUtils.update_class_mapping(label=action, data_path=save_path)
        for n in range(frame_count):
            _, frame = cap.read()
            out.write(frame)
            results = self.mp_tracker.Detect(frame)
            frame = self.mp_tracker.draw_landmarks(frame=frame, results=results, face=False)
            frame = self._add_frame_texts(frame=frame, frame_text=f'STARTING COLLECTION FOR "{action}." f:{n}')
            cv.imshow(f'Capture Action {samp_num}', frame)
            cv.waitKey(1)
        cap.release()
        out.release()
        cv.destroyAllWindows()

    def _show_capture(self, cap, duration: int, frame_text, count_down: bool = True, samp_num: str = ''):
        fps = cap.get(cv.CAP_PROP_FPS)
        total_fps = int(fps * duration)
        for n in range(total_fps):
            color = (0, 255, 0)
            time_passed = n // fps
            f_frame_text = frame_text
            if count_down:
                count_down_str = f'  {duration - time_passed}'
                if n >= (total_fps - 8):
                    count_down_str = ' START!'
                    color = (0, 0, 255)
                f_frame_text = frame_text + count_down_str

            _, frame = cap.read()
            self._add_frame_texts(frame=frame, frame_text=f_frame_text, color=color)
            cv.imshow(f'Capture Action {samp_num}', frame)
            cv.waitKey(1)
        return cap

    def _add_frame_texts(self, frame, frame_text: str, pos=(12, 25), font=cv.FONT_HERSHEY_SIMPLEX, font_scale=1,
                         color=(0, 0, 255), thickness=2, line_type=cv.LINE_AA):
        cv.putText(frame, frame_text, pos, font, font_scale, color, thickness, line_type)
        return frame
