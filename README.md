This repo contains a simple program that can be used to manually check the ultrasound distance and camera data collected by the setup for the cycling safety project.

## Instructions for Using the Frame-by-Frame Checking Function

1. **Clone Repository:**
   - Clone this GitHub repository to your local machine using either of the following methods:
     - Use the command `git clone https://github.com/jhy9968/cs_data_alignment.git` in your terminal.
     - Alternatively, download the .zip file and extract its contents.

2. **Data Placement:**
   - Place your data into the "data" folder. Please adhere to the instructions provided within the "data" folder.

3. **Run Program:**
   - Open a terminal within the "cs_data_alignment" folder, then execute the following command:
     ```
     python .\interactive_frame_check.py -f [change this to data name] -ffc [1 or 0]
     ```
     - '-f [change this to data name]'
       - Replace "[change this to data name]" with the name of your data folder.
     - '-ffc [1 or 0]'
       - Set '-ffc 1' if you want to filp the front camera. Otherwise, set '-ffc 0'
       - By default '-ffc 0'

4. **Interface Operation:**
   - An interface will appear. The left panels display ultrasound distance measurements, while the right panels show corresponding camera frames at the current timestamp. To operate:
     - Move the mouse to any distance measurement point you're interested in, and left-click to jump to that specific timestamp.
     - Press 'a' to jump to the last timestamp.
     - Press 'd' to jump to the next timestamp.
     - Press 'e' to automatically play the data frame by frame.
     - Press the space bar to stop autoplay.
     - Press 'q' to quit.
