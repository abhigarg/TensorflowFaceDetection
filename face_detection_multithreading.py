import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import cv2
import time
import argparse
import numpy as np
import tensorflow as tf
from align import detect_and_align
from queue import Queue
from threading import Thread
from utils.app_utils import FPS, WebcamVideoStream, draw_boxes_and_labels
from object_detection.utils import label_map_util

from numba import cuda

CWD_PATH = os.getcwd()

# Path to frozen detection graph. This is the actual model that is used for the object detection.
MODEL_NAME = 'FACE DETECTION MODEL PB FILENAME'
MODEL_DIR = 'ENTER PB FILE DIR HERE'
PATH_TO_CKPT = os.path.join(MODEL_DIR, MODEL_NAME)

def detect_objects(image_np, sess, detection_graph):
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image_np, axis=0)
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

    # Each box represents a part of the image where a particular object was detected.
    boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

    # Each score represent how level of confidence for each of the objects.
    # Score is shown on the result image, together with the class label.
    scores = detection_graph.get_tensor_by_name('detection_scores:0')
    classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    # Actual detection.
    (boxes, scores, classes, num_detections) = sess.run(
        [boxes, scores, classes, num_detections],
        feed_dict={image_tensor: image_np_expanded})

    # Visualization of the results of a detection.
    rect_points, class_names, class_colors = draw_boxes_and_labels(
        boxes=np.squeeze(boxes),
        classes=np.squeeze(classes).astype(np.int32),
        scores=np.squeeze(scores),
        category_index=category_index,
        min_score_thresh=.5
    )
    return dict(rect_points=rect_points, class_names=class_names, class_colors=class_colors)


def worker(input_q, output_q):
    # Load a (frozen) Tensorflow model into memory.
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

        sess = tf.Session(graph=detection_graph)
        mtcnn = detect_and_align.create_mtcnn(sess, None)

    fps = FPS().start()
    while True:
        fps.update()
        frame = input_q.get()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)        
        face_patches, padded_bounding_boxes, landmarks = detect_and_align.detect_faces(frame_rgb, mtcnn)
        output = dict(face_boxes=padded_bounding_boxes)
        output_q.put(output)

    fps.stop()
    sess.close()
    cuda.select_device(0)
    cuda.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-src', '--source', dest='video_source', type=int,
                        default=0, help='Device index of the camera.')
    parser.add_argument('-wd', '--width', dest='width', type=int,
                        default=1920, help='Width of the frames in the video stream.')   #480
    parser.add_argument('-ht', '--height', dest='height', type=int, 
                        default=1080, help='Height of the frames in the video stream.') #360
    parser.add_argument('-model', '--path_to_model', dest='model_path', type=str, 
                        default='20180408-102900.pb', help='Path to FaceNet Model')

    args = parser.parse_args()    
    PATH_TO_CKPT = args.model_path

    if not os.path.exists(PATH_TO_CKPT):
        print("Model file not found. Enter values to MODEL_NAME, MODEL_DIR")
        sys.exit(1)

    input_q = Queue(5)  # fps is better if queue is higher but then more lags
    output_q = Queue()
    for i in range(1):
        t = Thread(target=worker, args=(input_q, output_q))
        t.daemon = True
        t.start()

    video_capture = WebcamVideoStream(src=args.video_source, width=args.width, height=args.height).start()    
    fps = FPS().start()
    frame_count = 0
    
    while True:
        frame_count += 1        
        frame = video_capture.read()        
        if frame_count % 5 == 0:            
            input_q.put(frame)        

        if output_q.empty():
            pass  # fill up queue
        else:
            font = cv2.FONT_HERSHEY_SIMPLEX
            data = output_q.get()
            face_boxes = data['face_boxes']

            for bb in face_boxes:
                cv2.rectangle(frame, (bb[0], bb[1]), (bb[2], bb[3]), (122, 244, 66), 4)                        
            cv2.imshow('Video', frame)
        fps.update()        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    fps.stop()    
    print('[INFO] elapsed time (total): {:.2f}'.format(fps.elapsed()))
    print('[INFO] approx. FPS: {:.2f}'.format(fps.fps()))



    video_capture.stop()
    cv2.destroyAllWindows()
