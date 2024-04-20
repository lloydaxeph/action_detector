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
