import os
import cv2
import argparse
import keyboard
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import medfilt
from matplotlib.patches import Rectangle


class Frame_check():
    def __init__(self, f, ffc) -> None:
        self.folder_path = 'Z:\\CS_data\\{}\\'.format(f)
        self.ffc = ffc
        self.auto_play = False

        # Frame rate (Hz) for ultrasound distance sensor
        self.distance_frame_rate = 100

        self.right_distance_data = pd.read_csv(os.path.join(self.folder_path, 'Distance/right_distances.txt'), delimiter=', ', header=None, engine='python')
        self.rear_distance_data = pd.read_csv(os.path.join(self.folder_path, 'Distance/rear_distances.txt'), delimiter=', ', header=None, engine='python')
        self.right_time_stamp = self.right_distance_data.iloc[:, 0].to_numpy()
        self.rear_time_stamp = self.rear_distance_data.iloc[:, 0].to_numpy()
        self.right_dist = self.right_distance_data.iloc[:, 1].to_numpy()
        self.rear_dist = self.rear_distance_data.iloc[:, 1].to_numpy()

        # # Process data
        # self.processed_right_dist = self.calculate_moving_average(self.right_dist, 10)
        # self.processed_right_dist = self.calculate_median_filter(self.processed_right_dist, 29)
        self.right_dist[self.right_dist == 0] = 1200
        self.rear_dist[self.rear_dist == 0] = 1200

        # Get file_names for front and rear image files
        self.front_image_folder_path = os.path.join(self.folder_path, 'Images/Front')
        self.front_file_names = os.listdir(self.front_image_folder_path)
        self.front_file_names = [f for f in self.front_file_names if f.endswith('.jpg')]

        self.rear_image_folder_path = os.path.join(self.folder_path, 'Images/Rear')
        self.rear_file_names = os.listdir(self.rear_image_folder_path)
        self.rear_file_names = [f for f in self.rear_file_names if f.endswith('.jpg')]

        # Initialise front and rear image filename
        self.front_image = None
        self.rear_image = None

        # Set anchor for frame checking ("right" or "rear")
        self.anchor = "right"

        # Initialise plots
        self.fig, self.axs = plt.subplots(2, 2)
        # plt.ion()

        # Change x unit to seconds
        self.x_right = np.arange(len(self.right_dist))/self.distance_frame_rate
        self.x_rear = np.arange(len(self.rear_dist))/self.distance_frame_rate

        # Plot the first line
        self.axs[0, 0].plot(self.x_right, self.right_dist)
        self.axs[0, 0].set_xlabel('Time (s)')
        self.axs[0, 0].set_ylabel('Distance (cm)')
        if self.right_dist[0] == 1200:
            self.right_title = self.axs[0, 0].set_title(f'Right distance: Out of range')
        else:
            self.right_title = self.axs[0, 0].set_title(f'Right distance: {self.right_dist[0]}cm = {self.right_dist[0]/100}m')

        # Plot the second line
        self.axs[1, 0].plot(self.x_rear, self.rear_dist)
        self.axs[1, 0].set_xlabel('Time (s)')
        self.axs[1, 0].set_ylabel('Distance (cm)')
        if self.rear_dist[0] == 1200:
            self.rear_title = self.axs[1, 0].set_title(f'Rear distance: Out of range')
        else:
            self.rear_title = self.axs[1, 0].set_title(f'Rear distance: {self.rear_dist[0]}cm = {self.rear_dist[0]/100}m')

        # Add cursor markers
        # Initial position for marker in right dist plot
        self.marker_right, = self.axs[0, 0].plot(self.x_right[0], self.right_dist[0], marker='o', color='red', markersize=5)
        # Initial position for marker in rear dist plot
        self.marker_rear, = self.axs[1, 0].plot(self.x_rear[0], self.rear_dist[0], marker='o', color='red', markersize=5)

        # Show image
        self.right_index = 0
        self.rear_index = 0
        front_image, rear_image = self.find_image(self.right_time_stamp[self.right_index])
        self.show_images(front_image, rear_image)

        # Title for images
        self.axs[0, 1].set_title('Front camera image')
        self.axs[1, 1].set_title('Rear camera image')

        plt.tight_layout()  # Adjust layout to prevent overlap

        # Connect the event handler function to the 'button_press_event' event
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)

        # Connect key press event to the figure
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)

        plt.show()


    def calculate_median_filter(self, data, window_size):
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


    def calculate_moving_average(self, data, window_size):
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


    def find_image(self, distance_stamp):
        '''
        Input: Ultrasound distance time stamp (int)
        Return: Corresponding image file name (str)
        '''
        print(f"Distance time stamp:\t{distance_stamp}")
        front_image = min(self.front_file_names, key=lambda x:abs(int(''.join(filter(str.isdigit, x)))-int(distance_stamp)))
        rear_image = min(self.rear_file_names, key=lambda x:abs(int(''.join(filter(str.isdigit, x)))-int(distance_stamp)))
        ctimestamp = ''.join(filter(str.isdigit, front_image))
        if front_image == self.front_image and rear_image == self.rear_image:
            self.marker_right.set_color('r')
            self.marker_rear.set_color('r')
            print("Same image")
            print(f'Time difference:\t{(int(ctimestamp)-int(distance_stamp))/1e9:.2f} seconds')
        else:
            self.show_images(front_image, rear_image)
            self.marker_right.set_color((0, 0.8, 0))
            self.marker_rear.set_color((0, 0.8, 0))
            self.front_image = front_image
            self.rear_image = rear_image
            print(f'Camera time stamp:\t{ctimestamp}')
            print(f'Time difference:\t{(int(ctimestamp)-int(distance_stamp))/1e9:.2f} seconds')
        return front_image, rear_image


    def show_images(self, front_image, rear_image):
        image_folder_path = os.path.join(self.folder_path, 'Images/Front')
        self.image1 = cv2.imread(os.path.join(image_folder_path, front_image))
        
        image_folder_path = os.path.join(self.folder_path, 'Images/Rear')
        self.image2 = cv2.imread(os.path.join(image_folder_path, rear_image))

        # Flip front camera
        if self.ffc:
            self.image1 = cv2.flip(cv2.flip(self.image1, 1), 0)

        self.axs[0, 1].cla()  # Clear axis
        self.axs[1, 1].cla()  # Clear axis

        # Convert BGR to RGB for display
        self.image1 = cv2.cvtColor(self.image1, cv2.COLOR_BGR2RGB)
        self.image2 = cv2.cvtColor(self.image2, cv2.COLOR_BGR2RGB)

        # Show images
        self.axs[0, 1].imshow(self.image1)
        self.axs[0, 1].axis('off')

        self.axs[1, 1].imshow(self.image2)
        self.axs[1, 1].axis('off')

        # Add lidar box
        self.show_lidar_box(self.image2.shape)

        # Title for images
        self.axs[0, 1].set_title('Front camera image')
        self.axs[1, 1].set_title('Rear camera image')

    
    def show_lidar_box(self, imshape):
        center_x = 722.5
        center_y = 311
        box_width = 66
        box_height = 90
        box = Rectangle((center_x - box_width/2, center_y - box_height/2), box_width, box_height, fill=False, edgecolor='r')
        self.axs[0, 1].add_patch(box)

        center_x = 1836
        center_y = 280
        box_width = 66
        box_height = 90
        box = Rectangle((center_x - box_width/2, center_y - box_height/2), box_width, box_height, fill=False, edgecolor='r')
        self.axs[0, 1].add_patch(box)
    

    def update_markers(self):
        # For right marker
        y_clicked = self.right_dist[self.right_index]
        self.marker_right.set_data([self.x_right[self.right_index]], [y_clicked])
        if y_clicked == 1200:
            self.right_title.set_text(f'Right distance: Out of range')
        else:
            self.right_title.set_text(f'Right distance: {y_clicked}cm = {y_clicked/100}m')
        # For rear marker
        y_clicked = self.rear_dist[self.rear_index]
        self.marker_rear.set_data([self.x_rear[self.rear_index]], [y_clicked])
        if y_clicked == 1200:
            self.rear_title.set_text(f'Rear distance: Out of range')
        else:
            self.rear_title.set_text(f'Rear distance: {y_clicked}cm = {y_clicked/100}m')


    def on_click(self, event):
        if event.inaxes == self.axs[0, 0]:
            self.anchor = "right"
            x_clicked = event.xdata
            print(f"\nClicked at:\t\t{x_clicked:.2f} seconds")
            # Update index
            self.right_index = np.argmin(np.abs(self.x_right - x_clicked))
            self.anchor_time_stamp = self.right_time_stamp[self.right_index]
            self.rear_index = np.argmin(np.abs(self.rear_time_stamp - self.anchor_time_stamp))
            # Update markers
            self.update_markers()
            plt.draw()
            # Show image
            front_image, rear_image = self.find_image(self.right_time_stamp[int(x_clicked*self.distance_frame_rate)])
            # if front_image == self.front_image and rear_image == self.rear_image:
            #     self.marker_right.set_color('r')
            #     self.marker_rear.set_color('r')
            #     print("Same image")
            # else:
            #     self.marker_right.set_color((0, 0.8, 0))
            #     self.marker_rear.set_color((0, 0.8, 0))
            #     self.front_image = front_image
            #     self.rear_image = rear_image
            #     print('Camera time stamp:\t'+''.join(filter(str.isdigit, front_image)))
            #     self.show_images(front_image, rear_image)
        elif event.inaxes == self.axs[1, 0]:
            self.anchor = "rear"
            x_clicked = event.xdata
            print(f"\nClicked at:\t\t{x_clicked:.2f} seconds")
            # Update index
            self.rear_index = np.argmin(np.abs(self.x_rear - x_clicked))
            self.anchor_time_stamp = self.rear_time_stamp[self.rear_index]
            self.right_index = np.argmin(np.abs(self.right_time_stamp - self.anchor_time_stamp))
            # Update markers
            self.update_markers()
            plt.draw()
            # Show image
            front_image, rear_image = self.find_image(self.rear_time_stamp[int(x_clicked*self.distance_frame_rate)])
            # if front_image == self.front_image and rear_image == self.rear_image:
            #     self.marker_right.set_color('r')
            #     self.marker_rear.set_color('r')
            #     print("Same image")
            # else:
            #     self.show_images(front_image, rear_image)
            #     self.marker_right.set_color((0, 0.8, 0))
            #     self.marker_rear.set_color((0, 0.8, 0))
            #     self.front_image = front_image
            #     self.rear_image = rear_image
            #     print('Camera time stamp:\t'+''.join(filter(str.isdigit, front_image)))


    def update_plot(self):
        self.update_markers()
        plt.draw()
        if self.anchor == "right":
            front_image, rear_image = self.find_image(self.right_time_stamp[int(self.x_rear[self.right_index]*self.distance_frame_rate)])
        else:
            front_image, rear_image = self.find_image(self.rear_time_stamp[int(self.x_rear[self.rear_index]*self.distance_frame_rate)])
        # if front_image == self.front_image and rear_image == self.rear_image:
        #     self.marker_right.set_color('r')
        #     self.marker_rear.set_color('r')
        #     print("Same image")
        # else:
        #     self.marker_right.set_color((0, 0.8, 0))
        #     self.marker_rear.set_color((0, 0.8, 0))
        #     self.front_image = front_image
        #     self.rear_image = rear_image
        #     print('Camera time stamp:\t'+''.join(filter(str.isdigit, front_image)))
        #     self.show_images(front_image, rear_image)


    def on_key_press(self, event):
        if event.key == 'q':
            plt.close()  # Close the figure if 'q' is pressed
        elif event.key == 'a' or event.key == 'd':
            if event.key == 'a':
                self.right_index -= 1
                self.rear_index -= 1
            else:
                self.right_index += 1
                self.rear_index += 1
            print('\n')
            self.update_plot()
        elif event.key == 'e':
            print('Play')
            self.auto_play_frame()


    def auto_play_frame(self):
        global stop_flag
        stop_flag = False
        keyboard.on_press(on_space_pressed)

        while not stop_flag:
            self.right_index += 1
            self.rear_index += 1
            self.update_plot()
            plt.show(block=False)
            plt.pause(1/self.distance_frame_rate)

        keyboard.unhook_all()  # Unhook the keyboard listener when done


def on_space_pressed(event):
        if event.name == 'space' or event.name == 'q':
            print("Stop")
            global stop_flag
            stop_flag = True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="For folder name")
    parser.add_argument("-f", "--f", type=str, help="Folder name")
    parser.add_argument("-ffc", "--ffc", type=int, default=0, help="Flip front camera. To flip the front camera, -ffc 1. Otherwise, -ffc 0")
    args = parser.parse_args()

    # Start a frame check
    frame_check = Frame_check(args.f, args.ffc)