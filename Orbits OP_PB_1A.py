import cv2
import numpy as np
img_1 = cv2.imread('csv_data/pic1.jpg')
img1 = cv2.resize(img_1, (500, 500), interpolation=cv2.INTER_CUBIC)
# apply canny filter for edge detection, and highlight the edges
img1 = cv2.GaussianBlur(img1, (5, 5), 0)
edge = cv2.Canny(img1, 200, 255)
cv2.imshow('edge', edge)
# find contours in the image
contours, _ = cv2.findContours(edge, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# draw contours on the image
for i in range(len(contours)):
    cv2.drawContours(img1, contours, i, (0, 255, 0), 1)
# draw contours on the image
for i in range(len(contours)):
    cv2.drawContours(img1, contours, i, (0, 255, 0), 1)
cv2.imshow('contours', img1)
cv2.waitKey(0)
cv2.destroyAllWindows()
img_2 = cv2.imread('csv_data/pic2.jpg', cv2.IMREAD_GRAYSCALE)
img2 = cv2.resize(img_2, (500, 500), interpolation=cv2.INTER_CUBIC)
cv2.imshow('contours', img2)
cv2.waitKey(0)
cv2.destroyAllWindows()
# apply canny filter for edge detection, and highlight the edges
img2 = cv2.GaussianBlur(img2, (5,5), 0)
# Step 3: Edge Detection using Canny with Otsu’s thresholding
high_thresh, _ = cv2.threshold(img2, 0, 255, cv2.THRESH_OTSU)  # Otsu’s method
low_thresh = 0.5 * high_thresh
edges = cv2.Canny(img2, 250, 255)
# Step 4: Crater Detection using Hough Circle Transform
circles = cv2.HoughCircles(img2, cv2.HOUGH_GRADIENT, dp=1.2, minDist=25,
                           param1=high_thresh, param2=25, minRadius=15, maxRadius=100)

# Convert grayscale image to BGR for visualization
output = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
# Draw detected craters
if circles is not None:
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        cv2.circle(output, (i[0], i[1]), i[2], (0, 255, 0), 2)  # Draw crater
        cv2.circle(output, (i[0], i[1]), 2, (0, 0, 255), 2)  # Draw center

print("Detected Craters:")
cv2.imshow('new_image', output)  # Display in Colab
cv2.waitKey(0)
cv2.destroyAllWindows()

