import cv2
import os
from tqdm import tqdm
import tkinter as tk

def create_video_from_images(image_folder, output_video_path, fps=30, flip=False):
    # Get the list of image filenames
    image_files = sorted(os.listdir(image_folder))
    image_files = [f for f in image_files if f.endswith('.jpg')]

    # Get the dimensions of the first image
    img = cv2.imread(os.path.join(image_folder, image_files[0]))
    height, width, layers = img.shape

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Specify video codec
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Initialize progress bar
    progress_bar = tqdm(total=len(image_files), desc="Creating Video", unit="frame")

    # Loop through each image and write it to the video
    for image_file in image_files:
        img = cv2.imread(os.path.join(image_folder, image_file))
        if flip:
            img = cv2.flip(img, 0)
        video_writer.write(img)
        progress_bar.update(1)

    # Close the progress bar
    progress_bar.close()

    # Release the video writer
    video_writer.release()

    print(f"Video created successfully: {output_video_path}")


def play_two_videos_up_down_resized(video_path1, video_path2, target_frame_rate):
    # Open the video files
    cap1 = cv2.VideoCapture(video_path1)
    cap2 = cv2.VideoCapture(video_path2)

    # Check if the video files are opened successfully
    if not cap1.isOpened() or not cap2.isOpened():
        print("Error: Could not open video file(s).")
        return

    # Calculate the delay between frames based on the target frame rate
    delay = int(1000 / target_frame_rate)

    # Get screen dimensions
    root = tk.Tk()
    screen_width = root.winfo_screenwidth()//2
    screen_height = root.winfo_screenheight()
    print("Screen width:", screen_width)
    print("Screen height:", screen_height)
    root.destroy()
    # screen_width = 1366 
    # screen_height = 768

    # Read and display the videos frame by frame
    while True:
        ret1, frame1 = cap1.read()  # Read a frame from video 1
        ret2, frame2 = cap2.read()  # Read a frame from video 2

        if not ret1 or not ret2:
            break  # Break the loop if there are no more frames in either video

        # Resize frames to have the same width (assuming both videos have the same aspect ratio)
        frame_width = min(frame1.shape[1], frame2.shape[1])
        frame1 = cv2.resize(frame1, (frame_width, int(frame1.shape[0] * frame_width / frame1.shape[1])))
        frame2 = cv2.resize(frame2, (frame_width, int(frame2.shape[0] * frame_width / frame2.shape[1])))

        # Combine frames vertically
        combined_frame = cv2.vconcat([frame1, frame2])

        # Resize combined frame to fit within screen dimensions
        combined_frame = cv2.resize(combined_frame, (screen_width, int(screen_height / 2)))

        cv2.imshow('Two Videos', combined_frame)  # Display the combined frame
        if cv2.waitKey(delay) & 0xFF == ord('q'):
            break  # Break the loop if 'q' key is pressed

    # Release the video files and close the OpenCV window
    cap1.release()
    cap2.release()
    cv2.destroyAllWindows()




# Set the folder containing the images
image_folder = '//ad.monash.edu/home/User009/hjia0058/Desktop/CS_data/Box1_SD_1B/Images/'

# Specify frames per second (fps)
fps = 3

# Set the output video path
output_video_path_front = 'output_video_front.mp4'
# Call the function to create the video
# create_video_from_images(os.path.join(image_folder, 'Front'), os.path.join(image_folder, output_video_path_front), fps, flip=True)


# Set the output video path
output_video_path_rear = 'output_video_rear.mp4'
# Call the function to create the video
# create_video_from_images(os.path.join(image_folder, 'Rear'), os.path.join(image_folder, output_video_path_rear), fps)


# Play
play_two_videos_up_down_resized(os.path.join(image_folder, output_video_path_front), os.path.join(image_folder, output_video_path_rear), fps)