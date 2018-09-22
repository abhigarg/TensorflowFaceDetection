# TensorflowFaceDetection

Multithreading is used to speed up face detection using Tensorflow MTCNN model

## Requirements
1. Tensorflow > 1.0    			
2. Fork and install Tensorflow Object Detection API: https://github.com/tensorflow/models/tree/master/research/object_detection  	
3. OpenCV using `pip install opencv-contrib-python`
4. Fork the new trained model (`20180408-102900`) for Face Detection and Recognition from https://github.com/arunmandal53/facematch

## Usuages
In this script the default video source is set to webcam with width and height as 1920 x 1080. You may change the video source by passing arguments while running the python script in console:  		

`python face_detection_multithreading.py --source < video_source > --width < width_of_video_frames > --height < height_of_video_frames > -- model < path_to_facenet_model_pb_file >`

Current script is made to run on CPU, to run on GPU comment line-2: `os.environ["CUDA_VISIBLE_DEVICES"] = "-1"`

Current script will resize frame width to 400 before passing to Face Detection session, if you want to change that then you have to make changes to the method *detect_faces* in *align* module

## Test Results
Observed > 40FPS for 1920 x 1080 webcam feed on 16GB CPU RAM and intel i7 processor





