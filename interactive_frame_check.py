import os
import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt
from scipy.signal import medfilt


class Frame_check():
    def __init__(self, f, ffc) -> None:
        self.folder_path = '/home/hjia0058/Documents/CS_data/{}/'.format(f)
        self.ffc = ffc
        self.auto_play = False

        # Frame rate (Hz) for ultrasound distance and camera
        self.distance_frame_rate = 10
        self.camera_frame_rate = 3

        self.distance_data = pd.read_csv(os.path.join(self.folder_path, 'Distance/distances.txt'), delimiter=', ', header=None)
        self.time_stamp = self.distance_data.iloc[:, 0].to_numpy()
        self.right_dist = self.distance_data.iloc[:, 1].to_numpy()
        self.dist_check = self.distance_data.iloc[:, 2].to_numpy()

        for i, check in enumerate(self.dist_check):
            if check == "OVER_LIMIT_ERR":
                self.right_dist[i] = 4500
            elif check == "UNDER_LIMIT_ERR":
                self.right_dist[i] = 30
            elif check == "CHECKSUM_ERR":
                self.right_dist[i] = -30
            elif check == "SERIAL_ERR":
                self.right_dist[i] = -100
            elif check == "NO_DATA_ERR":
                self.right_dist[i] = -200

        # Process data
        self.processed_right_dist = self.calculate_moving_average(self.right_dist, 10)
        self.processed_right_dist = self.calculate_median_filter(self.processed_right_dist, 29)

        # Get file_names for front and rear image files
        self.front_image_folder_path = os.path.join(self.folder_path, 'Images/Rear')
        self.front_file_names = os.listdir(self.front_image_folder_path)
        self.front_file_names = [f for f in self.front_file_names if f.endswith('.jpg')]

        self.rear_image_folder_path = os.path.join(self.folder_path, 'Images/Front')
        self.rear_file_names = os.listdir(self.rear_image_folder_path)
        self.rear_file_names = [f for f in self.rear_file_names if f.endswith('.jpg')]

        # Initialise front and rear image filename
        self.front_image = None
        self.rear_image = None

        # Initialise plots
        self.fig, self.axs = plt.subplots(2, 2)

        # Change x unit to seconds
        self.x = np.arange(len(self.right_dist))/self.distance_frame_rate

        # Plot the first line
        self.axs[0, 0].plot(self.x, self.right_dist)
        self.axs[0, 0].set_xlabel('Time (s)')
        self.axs[0, 0].set_ylabel('Distance (mm)')
        self.axs[0, 0].set_title('Original right distance')

        # Plot the second line
        self.axs[1, 0].plot(self.processed_right_dist)
        self.axs[1, 0].set_ylabel('Distance (mm)')
        self.axs[1, 0].set_title('Processed right distance')

        # Show image
        self.index = 0
        front_image, rear_image = self.find_image(self.time_stamp[self.index])
        self.show_images(front_image, rear_image)

        # Title for images
        self.axs[0, 1].set_title('Front camera image')
        self.axs[1, 1].set_title('Rear camera image')

        # Add cursor marker
        self.x_init, self.y_init = self.x[0], self.right_dist[0]  # Initial position for marker
        self.marker, = self.axs[0, 0].plot(self.x_init, self.y_init, marker='o', color='red', markersize=5)

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
        front_image = min(self.front_file_names, key=lambda x:abs(int(''.join(filter(str.isdigit, x)))-int(distance_stamp)))
        rear_image = min(self.rear_file_names, key=lambda x:abs(int(''.join(filter(str.isdigit, x)))-int(distance_stamp)))
        # print(front_image, rear_image)
        return front_image, rear_image
        

    def show_images(self, front_image, rear_image):
        image_folder_path = os.path.join(self.folder_path, 'Images/Rear')
        self.image1 = plt.imread(os.path.join(image_folder_path, front_image))
        image_folder_path = os.path.join(self.folder_path, 'Images/Front')
        self.image2 = plt.imread(os.path.join(image_folder_path, rear_image))

        # Flip front camera
        if self.ffc:
            self.image1 = np.flipud(np.fliplr(self.image1))

        # Show images
        self.axs[0, 1].imshow(self.image1)
        self.axs[0, 1].axis('off')  # Hide axes

        self.axs[1, 1].imshow(self.image2)
        self.axs[1, 1].axis('off')  # Hide axes

        # Title for images
        self.axs[0, 1].set_title('Front camera image')
        self.axs[1, 1].set_title('Rear camera image')
    

    def on_click(self, event):
        if event.inaxes == self.axs[0, 0]:
            x_clicked = event.xdata
            print(f"Clicked on x = {x_clicked}")
            # Update marker position
            self.index = np.argmin(np.abs(self.x - x_clicked))
            y_clicked = self.right_dist[self.index]
            self.marker.set_data(self.x[self.index], y_clicked)
            plt.draw()
            # Show image
            front_image, rear_image = self.find_image(self.time_stamp[int(x_clicked*self.distance_frame_rate)])
            self.show_images(front_image, rear_image)


    def on_key_press(self, event):
        if event.key == 'q':
            plt.close()  # Close the figure if 'q' is pressed
        elif event.key == 'a':
            self.index -= 1
            y_clicked = self.right_dist[self.index]
            self.marker.set_data(self.x[self.index], y_clicked)
            plt.draw()
            front_image, rear_image = self.find_image(self.time_stamp[int(self.x[self.index]*self.distance_frame_rate)])
            if front_image == self.front_image:
                self.marker.set_color('r')
                print("Same image")
            else:
                self.marker.set_color((0, 0.8, 0))
                self.front_image = front_image
                self.rear_image = rear_image
                print(front_image, rear_image)
                self.axs[0, 1].cla()  # Clear axis
                self.axs[1, 1].cla()  # Clear axis
                self.show_images(front_image, rear_image)
        elif event.key == 'd':
            self.index += 1
            y_clicked = self.right_dist[self.index]
            self.marker.set_data(self.x[self.index], y_clicked)
            plt.draw()
            front_image, rear_image = self.find_image(self.time_stamp[int(self.x[self.index]*self.distance_frame_rate)])
            if front_image == self.front_image:
                self.marker.set_color('r')
                print("Same image")
            else:
                self.marker.set_color((0, 0.8, 0))
                self.front_image = front_image
                self.rear_image = rear_image
                print(front_image, rear_image)
                self.axs[0, 1].cla()  # Clear axis
                self.axs[1, 1].cla()  # Clear axis
                self.show_images(front_image, rear_image)
    #     elif event.key == ' ':
    #         if not self.auto_play:
    #             self.auto_play = True
    #             self.auto_play_frame()
    #         else:
    #             self.auto_play = False

    # def auto_play_frame(self):
    #     while self.auto_play:
    #         self.index += 1
    #         y_clicked = self.right_dist[self.index]
    #         self.marker.set_data(self.x[self.index], y_clicked)
    #         plt.draw()
    #         front_image, rear_image = self.find_image(self.time_stamp[int(self.x[self.index]*self.distance_frame_rate)])
    #         if front_image == self.front_image:
    #             self.marker.set_color('r')
    #             print("Same image")
    #         else:
    #             self.marker.set_color((0, 0.8, 0))
    #             self.front_image = front_image
    #             self.rear_image = rear_image
    #             print(front_image, rear_image)
    #             self.axs[0, 1].cla()  # Clear axis
    #             self.axs[1, 1].cla()  # Clear axis
    #             self.show_images(front_image, rear_image)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="For folder name")
    parser.add_argument("-f", "--f", type=str, help="Folder name")
    parser.add_argument("-ffc", "--ffc", type=int, default=0, help="Flip front camera. To flip the front camera, -ffc 1. Otherwise, -ffc 0")
    args = parser.parse_args()

    frame_check = Frame_check(args.f, args.ffc)