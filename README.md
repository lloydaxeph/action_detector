# Live Human Action Detector (Sign Language Detector)

## 1.0 About
A simple model that will detect specific human actions with just a few samples for training. Intended to be used as a live sign language translator.
<br/>
![demo](https://user-images.githubusercontent.com/67902015/152925333-8fdfeab3-2218-4196-b091-82208350bdf6.gif)
<br/>

### 1.1 Model
Using [MediaPipe's Holistic Model](https://developers.google.com/mediapipe/solutions/vision/holistic_landmarker), we can acquire the specific keypoints in a human subjects Hands, Shoulders and Face.
These keypoints are the input in our Convolutional Neural Network (CNN) model. The model structure is just very basic as shown below:
<br/>
![MediaPipe-Holistic-API-a-Hand-b-Pose](https://github.com/lloydaxeph/human_action_detector/assets/158691653/d525f98a-fd3e-424f-8e5b-2707f0583a5c)<br/>

![Convolution Layers](https://github.com/lloydaxeph/human_action_detector/assets/158691653/39a55196-1e84-48a2-a889-1549ccfa6a0b)<br/>

## 2.0 Getting Started
### 2.1 Installation
Install the required packages
```
pip3 install -r requirements.txt
```
### 2.2 Capture Data
Using <b>utils.CaptureUtils</b>, you can automatically capture video data with specific parameters for training this model. Note that all training data should be in a single directory. And each video file should follow this filename format: {class}-{data_id}.mp4. <b>CaptureUtils</b> will automatically do this for you.
```python
cap = CaptureUtils()
for n in range(num_sample):
  cap.capture_action(action=action, frame_count=frame_count, save_path=data_save_path, samp_num=str(n + 1))
```

### 2.3 Create a Custom Dataset
Creation of the Custom Dataset is very straight forward. You can use the following code for reference:
```python
dataset = CustomDataset(data_path=data_save_path, save_dataset=is_save_data, preprocessed_dataset_path=preprocessed_dataset_path)
```

### 2.4 Training
You can create and train a <b>ActionDetector model</b> using the following code:
```python
detector = ActionDetector(class_mapping=dataset.class_mapping, frames_interval=dataset.frames_interval, include_pose=dataset.include_pose, pose_positions=dataset.pose_positions,
  model_type=model_type, initial_hidden_layer=initial_hidden_layer)
detector.train(train_data=train_ds, validation_data=val_ds, epochs=epochs, batch_size=batch_size, early_stopping_patience=early_stopping_patience, test=is_test, test_data=test_ds)
```

### 2.5 Run
To use the model, you can use <b>ActionDetector.run</b> function.
```python
detector.run()
```

For the complete demonstration, you can follow the [sample_implementation.py](https://github.com/lloydaxeph/human_action_detector/blob/main/sample_implementation.py) script above.
