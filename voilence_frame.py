import cv2
import numpy as np
from collections import deque

import cv2
import numpy as np
from collections import deque

import os
import shutil
import cv2
import math
import random
import numpy as np
import datetime as dt
import tensorflow 
import keras
from collections import deque
import matplotlib.pyplot as plt
# plt.style.use("seaborn")

# %matplotlib inline
from sklearn.model_selection import train_test_split
from keras.layers import *
from keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from tensorflow.keras.utils import plot_model

# #importing the model
# import pickle
# #saving the model
# model_pkl_file = "model.pkl"  
# with open(model_pkl_file, 'rb') as file:  
#     final_model = pickle.load(file)
    
IMAGE_HEIGHT , IMAGE_WIDTH = 64, 64

import boto3
from botocore.exceptions import NoCredentialsError
import datetime
import uuid

def send_violence_frame_to_s3(local_save_path):
    try:
        # Create an S3 client
        s3_object_key = f'violence_frames/{str(uuid.uuid4())}.jpg'  # Unique identifier in the object key
        bucket_name = 'cloud-project-one'
        s3 = boto3.client('s3')

        # Upload the file
        s3.upload_file(local_save_path, bucket_name, s3_object_key)

        print(f"File uploaded successfully to {bucket_name}/{s3_object_key}")

    except NoCredentialsError:
        print("Credentials not available")
    

def real_time_detection2(output_file_path, sequence_length, model, class_list, violence_output_path, local_save_path):
    
    print("hiiiiiiiiiiiiiiiii")

    video_reader = cv2.VideoCapture(1)

    if not video_reader.isOpened():
        print("Error: Could not open the camera.")
        return

    frame_width = int(video_reader.get(3))
    frame_height = int(video_reader.get(4))

    video_writer = cv2.VideoWriter(output_file_path, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'),
                                      30, (frame_width, frame_height))  # 30 FPS

    violence_writer = None
    frames_queue = deque(maxlen=sequence_length)
    predicted_class_name = ''

    while True:
        ret, frame = video_reader.read()

        if not ret:
            break

        resized_frame = cv2.resize(frame, (IMAGE_WIDTH, IMAGE_HEIGHT))
        normalized_frame = resized_frame / 255.0

        frames_queue.append(normalized_frame)

        if len(frames_queue) == sequence_length:
            predicted_labels_probabilities = model.predict(np.expand_dims(frames_queue, axis=0))[0]
            predicted_label = np.argmax(predicted_labels_probabilities)
            predicted_class_name = class_list[predicted_label]

        # Display predicted class on frame
        if predicted_class_name == "Violence":
            cv2.putText(frame, predicted_class_name, (5, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 12)

            if violence_writer is None:
                violence_writer = cv2.VideoWriter(violence_output_path, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'),
                                                 30, (frame_width, frame_height))

            violence_writer.write(frame)

            # Save the frame locally
            cv2.imwrite(local_save_path, frame)

            # Add a condition to stop when violence is detected
            break

        else:
            cv2.putText(frame, predicted_class_name, (5, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 12)

        video_writer.write(frame)

        cv2.imshow("Real-Time Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_reader.release()
    video_writer.release()

    if violence_writer is not None:
        violence_writer.release()

    cv2.destroyAllWindows()
    send_violence_frame_to_s3(local_save_path)
    
    
    


# # Example usage:
# output_file_path = 'output_video.mp4'
# violence_output_path = 'violence_frames.mp4'
# sequence_length = 16  # Adjust as needed
# local_save_path = 'violence_frame.jpg'  # Adjust the format as needed
# model = final_model  # Replace with your model
# class_list = ['Non-Violence', 'Violence']  # Replace with your class list
# real_time_detection(output_file_path, sequence_length, model, class_list, violence_output_path, local_save_path)
