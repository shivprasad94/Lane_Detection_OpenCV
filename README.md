# Lane_Detection_OpenCV
The lines drawn on roads indicate to human drivers where the lanes are and act as a guiding reference to which direction to steer the vehicle accordingly and convention to how vehicle agents interact harmoniously on the road.

# Techinque used to Slove the problem 
Our first task is to detect prominent straight lines in the camera feed through edge detection and feature extraction techniques. We will be using OpenCV, an open source library of computer vision algorithms, for implementation.
 
- Take input image and convert it into grayscale because we only need the luminance channel for detecting edges - less computationally expensive :P
- Than Applies a 5x5 gaussian blur with deviation of 0 to frame - not mandatory since Canny will do this for us
- Applies Canny edge detector with minVal of 50 and maxVal of 150
- We will handcode a triangular mask to segment the lane area and discard the irrelevant areas in the frame.
- A triangular polygon for the mask defined by three (x, y) coordinates (0, height), (800, height), (380, 290)

![image](https://user-images.githubusercontent.com/13836633/100740280-74d8b100-33fe-11eb-995e-f5dd89700ccc.png)

- Hough transform
we can represent a straight line as y = mx + b by plotting y against x. However, we can also represent this line as a single point in Hough space by plotting b against m. For example, a line with the equation y = 2x + 1 may be represented as (2, 1) in Hough space.
Now, what if instead of a line, we had to plot a 'POINT' in the Cartesian coordinate system. There are many possible lines which can pass through this point, each line with different values for parameters m and b. These possible lines can be plotted in Hough space


# OpenCV image Transformation

![1_wVfCstrB_YLG_UZE3X5JLw](https://user-images.githubusercontent.com/13836633/100738882-4954c700-33fc-11eb-83d4-0ac42bf46b13.png)

# Output

![F26827C6-2CF0-46F7-B07A-631B7ECFDC53](https://user-images.githubusercontent.com/13836633/100738629-dcd9c800-33fb-11eb-85f3-afa7d0a8a5ef.GIF)
