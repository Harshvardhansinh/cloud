import streamlit as st
import cv2
import os
import shutil
import numpy as np

from model import do_prediction,send_data
IMAGE_HEIGHT , IMAGE_WIDTH = 64, 64

#importing the model
import pickle
#saving the model
model_pkl_file = "model.pkl"  
with open(model_pkl_file, 'rb') as file:  
    final_model = pickle.load(file)

from real_time import real_time_detection
from voilence_frame import real_time_detection2

# def main():
#     st.title("Violence Detection")
#     uploaded_video = st.file_uploader("Upload a video", type=["mp4", "avi"])

#     if uploaded_video is not None:
#         st.video(uploaded_video)

#         if st.button("Process Video"):
#             input_video_path = "input/input_video.mp4"  # Input video path

#             # Save the uploaded video to the input folder
#             with open(input_video_path, "wb") as f:
#                 f.write(uploaded_video.read())

#             # Process the video
#             process_video(input_video_path)


def main():
    st.title("Violence Detection")
    option = st.radio("Choose an option", ("Upload Video", "Live Camera","Detect violence frame"))

    if option == "Upload Video":
        uploaded_video = st.file_uploader("Upload a video", type=["mp4", "avi"])

        if uploaded_video is not None:
            st.video(uploaded_video)

            if st.button("Process Video"):
                input_video_path = "input/input_video.mp4"  # Input video path

                # Save the uploaded video to the input folder
                with open(input_video_path, "wb") as f:
                    f.write(uploaded_video.read())

                # Process the video
                process_video(input_video_path)

    elif option == "Live Camera":
        st.write("Live Camera")

        # Add a button to start capturing from the live camera
        if st.button("Start Live Camera"):
            sequence_length = 16 
            model = final_model 
            class_list = ['Non-Violence', 'Violence']  
            output_file_path = 'output_video.mp4'
            violence_output_path = 'violence_frames.mp4'
            local_save_path = 'violence_frame.jpg'  

            real_time_detection(output_file_path, sequence_length, model, class_list)
            
    
    elif option == "Detect violence frame":
        st.write("Detect violence frame")

        # Add a button to start capturing from the live camera
        if st.button("Detect voilence frame"):
            sequence_length = 16 
            model = final_model 
            class_list = ['Non-Violence', 'Violence']  
            output_file_path = 'output_video.mp4'
            violence_output_path = 'violence_frames.mp4'
            local_save_path = 'violence_frame.jpg'  

            real_time_detection2(output_file_path, sequence_length, model, class_list, violence_output_path, local_save_path)
            st.success("send detected frame to s3!!")
            
    
            
    
    

            
def process_video(input_video_path):
    do_prediction(input_video_path)
    
    print("detection is done !!!")
    st.success("Detection is done !!!")
    
    st.video(r"test_videos\Output-Test-Video.mp4")
    
    send_data()
    
    print("sent data!!")
    
    st.success("Frames sent to s3 !!")
    
    shutil.rmtree(r"output_frames")


if __name__ == "__main__":
    main()
