import os
import random
import shutil
import cv2
import math
import random
import numpy as np
import datetime as dt
from collections import deque
import matplotlib.pyplot as plt
import os
import boto3
from botocore.exceptions import NoCredentialsError
# plt.style.use("seaborn")


# Specify the height and width to which each video frame will be resized in our dataset.
IMAGE_HEIGHT , IMAGE_WIDTH = 64, 64

# Specify the number of frames of a video that will be fed to the model as one sequence.
SEQUENCE_LENGTH = 16

# DATASET_DIR = r"C:\Users\harsh\Desktop\CSE2026\Project\Real Life Violence Dataset"

CLASSES_LIST = ["NonViolence","Violence"]

    # Construct the output video path.
test_videos_directory = 'test_videos'
os.makedirs(test_videos_directory, exist_ok = True)

output_video_file_path = f'{test_videos_directory}/Output-Test-Video.mp4'

ax= plt.subplot()



#importing the model
import pickle
#saving the model
model_pkl_file = "model.pkl"  
with open(model_pkl_file, 'rb') as file:  
    final_model = pickle.load(file)
    
    
def predict_frames(video_file_path, output_file_path, SEQUENCE_LENGTH):
    # Read from the video file.
    video_reader = cv2.VideoCapture(video_file_path)

    # Get the width and height of the video.
    original_video_width = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_video_height = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # VideoWriter to store the output video in the disk.
    video_writer = cv2.VideoWriter(output_file_path, cv2.VideoWriter_fourcc(*'avc1'),
    video_reader.get(cv2.CAP_PROP_FPS), (original_video_width, original_video_height))

    # Declare a queue to store video frames.
    frames_queue = deque(maxlen=SEQUENCE_LENGTH)

    # Store the predicted class in the video.
    predicted_class_name = ''

    # Iterate until the video is accessed successfully.
    while video_reader.isOpened():
        ok, frame = video_reader.read()

        if not ok:
            break

        # Resize the Frame to fixed Dimensions.
        resized_frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))

        # Normalize the resized frame
        normalized_frame = resized_frame / 255

        # Appending the pre-processed frame into the frames list.
        frames_queue.append(normalized_frame)

        # We Need at Least number of SEQUENCE_LENGTH Frames to perform a prediction.
        # Check if the number of frames in the queue are equal to the fixed sequence length.
        if len(frames_queue) == SEQUENCE_LENGTH:

            # Pass the normalized frames to the model and get the predicted probabilities.
            predicted_labels_probabilities = final_model.predict(np.expand_dims(frames_queue, axis=0))[0]

            # Get the index of class with highest probability.
            predicted_label = np.argmax(predicted_labels_probabilities)

            # Get the class name using the retrieved index.
            predicted_class_name = CLASSES_LIST[predicted_label]

            # Write predicted class name on top of the frame.
            if predicted_class_name == "Violence":
                cv2.putText(frame, predicted_class_name, (5, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 12)
            else:
                cv2.putText(frame, predicted_class_name, (5, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 12)

            # Write The frame into the disk using the VideoWriter
            video_writer.write(frame)

    video_reader.release()
    video_writer.release()

plt.style.use("default")

# To show Random Frames from the saved output predicted video (output predicted video doesn't show on the notebook but can be downloaded)
def show_pred_frames(pred_video_path): 

    plt.figure(figsize=(20,15))

    video_reader = cv2.VideoCapture(pred_video_path)

    # Get the number of frames in the video.
    frames_count = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Get Random Frames from the video then Sort it
    random_range = sorted(random.sample(range (SEQUENCE_LENGTH , frames_count ), 12))
        
    for counter, random_index in enumerate(random_range, 1):
        
        plt.subplot(5, 4, counter)

        # Set the current frame position of the video.  
        video_reader.set(cv2.CAP_PROP_POS_FRAMES, random_index)
        
        ok, frame = video_reader.read() 

        if not ok:
            break 

        frame = cv2.cvtColor(frame , cv2.COLOR_BGR2RGB)

        plt.imshow(frame);ax.figure.set_size_inches(20,20);plt.tight_layout()
                            
    video_reader.release()
    
    




def create_frames(output_video_file_path):
    # Create a directory to store the frames
    output_frames_directory = 'output_frames'
    os.makedirs(output_frames_directory, exist_ok=True)

    # Path to the video file
    video_file_path = output_video_file_path  # Replace with the path to your video file

    # Open the video file
    cap = cv2.VideoCapture(video_file_path)

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) - 1)  # Get the total number of frames in the video

    # Choose 10 random time spots within the video
    random_time_spots = [int(random.uniform(0, frame_count)) for _ in range(10)]

    for time_spot in random_time_spots:
        cap.set(cv2.CAP_PROP_POS_FRAMES, time_spot)
        ret, frame = cap.read()

        if ret:
            # Save the frame as an image in the output directory
            frame_filename = os.path.join(output_frames_directory, f'frame_{time_spot:04d}.jpg')
            cv2.imwrite(frame_filename, frame)
        else:
            print(f"Failed to capture frame at time spot {time_spot}.")

    # Release the video capture object
    cap.release()
    print("frames created")



def do_prediction(input_video_path):
    # Specifying video to be predicted
    input_video_file_path = input_video_path

    # Perform Prediction on the Test Video.
    predict_frames(input_video_file_path, output_video_file_path, SEQUENCE_LENGTH)

    # Show random frames from the output video
    show_pred_frames(output_video_file_path)
    
    create_frames(output_video_file_path)
    
    
    
    

def send_data():
    # AWS S3 bucket details
    bucket_name = 'cloud-project-one'

    # Local directory containing the image frames
    local_frames_directory = 'output_frames'

    # Initialize the S3 client
    s3 = boto3.client('s3')

    # List all files in the local frames directory
    local_files = os.listdir(local_frames_directory)

    # Create a unique folder name for each upload
    import uuid
    folder_name = str(uuid.uuid4())

    # Upload each file to S3 with the new folder name
    for local_file in local_files:
        local_file_path = os.path.join(local_frames_directory, local_file)
        s3_object_key = f'frames/{folder_name}/{local_file}'  # Modify the key as needed

        try:
            s3.upload_file(local_file_path, bucket_name, s3_object_key)
            print(f"Uploaded {local_file} to S3 as {s3_object_key}")
        except NoCredentialsError:
            print("AWS credentials not available.")

    print(f"Frames from the local directory have been uploaded to S3.")





