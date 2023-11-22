import os
import boto3
from botocore.exceptions import NoCredentialsError

def send_data():
    # AWS S3 bucket details
    bucket_name = 'cloud-project-one'

    # Local directory containing the image frames
    local_frames_directory = 'output_frames'

    # Initialize the S3 client
    s3 = boto3.client('s3')

    # List all files in the local frames directory
    local_files = os.listdir(local_frames_directory)

    for local_file in local_files:
        local_file_path = os.path.join(local_frames_directory, local_file)
        s3_object_key = f'frames/{local_file}'  # Modify the key as needed

        try:
            s3.upload_file(local_file_path, bucket_name, s3_object_key)
            print(f"Uploaded {local_file} to S3 as {s3_object_key}")
        except NoCredentialsError:
            print("AWS credentials not available.")

    print("Frames from the local directory have been uploaded to S3.")
