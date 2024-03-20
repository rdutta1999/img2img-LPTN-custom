import cv2

# Read your BGR image
BGR_image = cv2.imread("OIP.jpeg")

# Convert to YCbCr
YCbCr_image = cv2.cvtColor(BGR_image, cv2.COLOR_BGR2YCR_CB)

# Now you can work with the YCbCr image
# For example, you can convert it back to BGR using:
# BGR_image_back = cv2.cvtColor(YCbCr_image, cv2.COLOR_YCR_CB2BGR)

print(YCbCr_image.shape, YCbCr_image.dtype)
print(YCbCr_image)