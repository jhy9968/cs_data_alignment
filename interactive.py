import os
import cv2
import tkinter as tk
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import medfilt


def calculate_median_filter(data, window_size):
    """
    Calculates the median filter of the input data using the specified window size.

    Parameters:
    - data: NumPy array or list, the input data to calculate the median filter.
    - window_size: int, the size of the median filter window.

    Returns:
    - filtered_data: NumPy array, the filtered data using the median filter.
    """

    # Apply median filter
    filtered_data = medfilt(data, kernel_size=window_size)

    return filtered_data


def calculate_moving_average(data, window_size):
    """
    Calculates the moving average of the input data using the specified window size.

    Parameters:
    - data: NumPy array or list, the input data to calculate the moving average.
    - window_size: int, the size of the moving average window.

    Returns:
    - moving_avg: NumPy array, the moving average of the input data.
    """

    # Calculate the moving average
    moving_avg = np.convolve(data, np.ones(window_size)/window_size, mode='valid')

    return moving_avg


def create_video(image_files, output_video_path, folder_path, frame_rate=10, flip=False):
    # Get the dimensions of the first image
    img = cv2.imread(os.path.join(folder_path, image_files[0]))
    height, width, layers = img.shape

    # Create a VideoWriter object
    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), frame_rate, (width, height))

    # Initialize tqdm to display progress bar
    progress_bar = tqdm(total=len(image_files), desc='Creating video', unit='image')

    # Loop through the image files and write each frame to the video
    for image_file in image_files:
        # Read the image
        image_path = os.path.join(folder_path, image_file)  # Construct image path
        frame = cv2.imread(image_path)
        if flip:
            frame = cv2.flip(frame, 0)

        # Resize the frame if necessary
        # frame = cv2.resize(frame, (frame_width, frame_height))

        # Write the frame to the video
        out.write(frame)

        # Update the progress bar
        progress_bar.update(1)

    # Close the progress bar
    progress_bar.close()

    # Release the VideoWriter object
    out.release()


def play_two_videos(video_path1, video_path2, target_frame_rate):
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


def create_clip(start_stamp, end_stamp):

    frame_rate = camera_frame_rate  # Desired frame rate

    image_folder_path = os.path.join(folder_path, 'Images/Front')
    file_names = os.listdir(image_folder_path)
    file_names = [f for f in file_names if f.endswith('.jpg')]
    frames = []
    for file_name in file_names:
        frame_stamp = int(''.join(filter(str.isdigit, file_name)))
        if frame_stamp >= start_stamp and frame_stamp <= end_stamp:
            frames.append(file_name)
    if len(frames) == 0:
        print('No corresponding video clips found.')
        return
    frames = sorted(frames)
    output_video_path_front = 'output_video_clip_front.mp4'  # Output video path
    create_video(frames, output_video_path_front, image_folder_path, frame_rate, flip=True)


    image_folder_path = os.path.join(folder_path, 'Images/Rear')
    file_names = os.listdir(image_folder_path)
    file_names = [f for f in file_names if f.endswith('.jpg')]
    frames = []
    for file_name in file_names:
        frame_stamp = int(''.join(filter(str.isdigit, file_name)))
        if frame_stamp >= start_stamp and frame_stamp <= end_stamp:
            frames.append(file_name)
    if len(frames) == 0:
        print('No corresponding video clips found.')
        return
    frames = sorted(frames)
    output_video_path_rear = 'output_video_clip_rear.mp4'  # Output video path
    create_video(frames, output_video_path_rear, image_folder_path, frame_rate)


    play_two_videos(output_video_path_front, output_video_path_rear, frame_rate)


folder_path = '//ad.monash.edu/home/User009/hjia0058/Desktop/CS_data/23_12_05_16_11/'

# Frame rate (Hz) for ultrasound distance and camera
distance_frame_rate = 10
camera_frame_rate = 3

distance_data = pd.read_csv(os.path.join(folder_path, 'Distance/distances.txt'), delimiter=', ', header=None)
time_stamp = distance_data.iloc[:, 0].to_numpy()
right_dist = distance_data.iloc[:, 1].to_numpy()
# back_dist = distance_data.iloc[:, 3].to_numpy()

# Process data
processed_right_dist = calculate_moving_average(right_dist, 10)
processed_right_dist = calculate_median_filter(processed_right_dist, 29)
# processed_back_dist = calculate_median_filter(back_dist, 9)

fig, axs = plt.subplots(2)

# Change x unit to seconds
x = np.arange(len(right_dist))/distance_frame_rate

# Plot the first line
axs[0].plot(x, right_dist)
axs[0].set_xlabel('Time (s)')
axs[0].set_ylabel('Distance (mm)')
axs[0].set_title('Original right distance')

# Plot the second line
axs[1].plot(processed_right_dist)
axs[1].set_ylabel('Distance (mm)')
axs[1].set_title('Processed right distance')

start_stamp = -1
end_stamp = -1

def on_click(event):
    global start_stamp
    global end_stamp
    if event.inaxes == axs[0]:
        x_clicked = event.xdata
        print(f"Clicked on x = {x_clicked}")
        if start_stamp < 0:
            start_stamp = x_clicked
        else:
            end_stamp = x_clicked
            print("Create clip for {0:.2f} seconds".format(end_stamp - start_stamp))
            start_stamp = time_stamp[int(start_stamp*distance_frame_rate)]
            end_stamp = time_stamp[int(end_stamp*distance_frame_rate)]
            create_clip(start_stamp, end_stamp)
            start_stamp = -1
            end_stamp = -1

# # Plot the first line
# axs[0].plot(right_dist)
# axs[0].set_title('Right distance')

# # Plot the second line
# axs[1].plot(back_dist)
# axs[1].set_title('Back distance')

plt.tight_layout()  # Adjust layout to prevent overlap

# Connect the event handler function to the 'button_press_event' event
fig.canvas.mpl_connect('button_press_event', on_click)

plt.show()
