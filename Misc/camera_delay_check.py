import os
import numpy as np
import pandas as pd


if __name__ == "__main__":

    folder_path = 'S:/R-MNHS-SPHPM-EPM-PHTrauma/SMSR-RSIF/Data/All Data - Has to be cleaned/Processed_Data/User_7/22_12_13_03_32/'

    # Frame rate (Hz) for ultrasound distance and camera
    distance_frame_rate = 10
    camera_frame_rate = 3

    distance_data = pd.read_csv(os.path.join(folder_path, 'Distance/distance.txt'), delimiter=', ', header=None)
    distance_init_stamp = int(distance_data.iloc[:, 0].to_numpy()[0])

    # Get file_names for front image files
    front_image_folder_path = os.path.join(folder_path, 'Images/Front/Right')
    front_file_names = os.listdir(front_image_folder_path)
    front_file_names = [f for f in front_file_names if f.endswith('.jpg')]
    camera_init_stamp = min([int(''.join(filter(str.isdigit, x))) for x in front_file_names])

    print("Delay is {0:.3f}s".format((camera_init_stamp - distance_init_stamp)/1e9))