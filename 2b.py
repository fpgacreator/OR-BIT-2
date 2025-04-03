
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image in grayscale
image = cv2.imread('2/deepspace1 (1).jpg', cv2.IMREAD_GRAYSCALE)

if image is None:
    print("Error: Could not load image.")
    exit()


# Step 2: Noise Reduction using Median Blur
blurred = cv2.medianBlur(image, 17)


edges = cv2.Canny(blurred, 250, 255)

# Convert grayscale image to BGR for visualization
output = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

# Detect circles using HoughCircles
circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp=1.2, minDist=30, param1=50, param2=30, minRadius=5, maxRadius=15)

if circles is not None:
    circles = np.uint16(np.around(circles))
    for circle in circles[0, :]:
        # Draw the outer circle
        cv2.circle(output, (circle[0], circle[1]), circle[2], (0, 255, 0), 2)
        # Draw the center of the circle
        cv2.circle(output, (circle[0], circle[1]), 2, (0, 0, 255), 3)

cv2.imshow("image", output)
cv2.waitKey(0)
cv2.destroyAllWindows()