import numpy
import cv2
import sys

# Pass the full path of the image as an argument to the program
image = cv2.imread(sys.argv[1])

# Convert the image to gray scale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Compute the Scharr Gradient magnitude of the images in both the X and Y directions
gradientX = cv2.Sobel(gray_image, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=-1)
gradientY = cv2.Sobel(gray_image, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=-1)

# Subtract the Y gradient from the X gradient
gradient_diff = cv2.subtract(gradientX, gradientY)

# On each element of the input array, the function convertScaleAbs performs three operations sequentially:
# scaling, taking an absolute value, conversion to an unsigned 8-bit type:
gradient_diff = cv2.convertScaleAbs(gradient_diff)

# Blurs an image using the normalized box filter.
blurred = cv2.blur(gradient_diff, (11, 11))

# If pixel value is greater than a threshold value, it is assigned one value (may be white),
# else it is assigned another value (may be black).
ret, threshold = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)

# The function constructs and returns the structuring element that can be further passed to createMorphologyFilter(),
# erode(), dilate() or morphologyEx() . But you can also construct an
# arbitrary binary mask yourself and use it as the structuring element.
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7))

# The function can perform advanced morphological transformations using an erosion and dilation as basic operations.
closed = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, kernel)

# The function erodes the source image using the specified structuring element that determines the shape of a pixel
# neighborhood over which the minimum is taken. The function supports the in-place mode. Erosion can be applied several
# ( iterations ) times. In case of multi-channel images, each channel is processed independently.
closed = cv2.erode(closed, None, iterations=20)

# The function dilates the source image using the specified structuring element that determines the shape of a pixel
# neighborhood over which the maximum is taken. The function supports the in-place mode. Dilation can be applied several
# ( iterations ) times. In case of multi-channel images, each channel is processed independently.
closed = cv2.dilate(closed, None, iterations=20)

# The function retrieves contours from the binary image using the algorithm Source image is modified by this function.
# Also, the function does not take into account 1-pixel border of the image (it's filled with 0's and used for neighbor
# analysis in the algorithm), therefore the contours touching the image border will be clipped.
img, contours, hierarchy = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw green borders over all the contours
for contour in contours:
    rectangle = cv2.minAreaRect(contour)
    box = numpy.int32(cv2.boxPoints(rectangle))
    cv2.drawContours(image, [box], -1, (0, 255, 0), 4)

# Display the Image
cv2.imshow("Image", image)
cv2.waitKey(0)
