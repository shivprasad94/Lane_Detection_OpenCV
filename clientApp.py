import cv2 as cv
import numpy as np

from laneDetectionWithImage import LaneDetectorOnImage
from laneDetectionWithVideo import LaneDetector

choice = input('Want to Execute on image or Video \n 1 for image \n 2 for video......  \n ')
if choice == '1':
    choice = input("Enter choice if you want to see all the output images as we execute the code Y/N")
    if choice.lower() == 'y':
        choice = True
    elif choice.lower() == 'n':
        choice = False
    obj = LaneDetectorOnImage()
    obj.executeProcessing(choice)

if choice == '2':
    laneDtctrObj = LaneDetector()
    # The video feed is read in as a VideoCapture object
    cap = cv.VideoCapture("InputVideo/input.mp4")
    # cap = cv.VideoCapture('challenge.mp4')
    while (cap.isOpened()):
        # ret = a boolean return value from getting the frame, frame = the current frame being projected in the video
        ret, frame = cap.read()
        canny = laneDtctrObj.cannyEdgeDetection(frame)
        cv.imshow("CannyEdge", canny)
        # plt.imshow(frame)
        # plt.show()
        segment = laneDtctrObj.performFrameSegmentation(canny)
        hough = cv.HoughLinesP(segment, 2, np.pi / 180, 100, np.array([]), minLineLength=100, maxLineGap=50)
        # Averages multiple detected lines from hough into one line for left border of lane and one line for right border of lane
        lines = laneDtctrObj.findTracks(frame, hough)
        # Visualizes the lines
        lines_visualize = laneDtctrObj.displayTrackLines(frame, lines)
        cv.imshow("Hough lines", lines_visualize)
        # Overlays lines on frame by taking their weighted sums and adding an arbitrary scalar value of 1 as the gamma argument
        output = cv.addWeighted(frame, 0.9, lines_visualize, 1, 1)
        # Opens a new window and displays the output frame
        cv.imshow("output", output)
        # Frames are read by intervals of 10 milliseconds. The programs breaks out of the while loop when the user presses the 'q' key
        if cv.waitKey(10) & 0xFF == ord('q'):
            break
    # The following frees up resources and closes all windows
    cap.release()
    cv.destroyAllWindows()