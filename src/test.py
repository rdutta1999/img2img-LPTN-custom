import cv2

# Load the image
img = cv2.imread('./car.png')
lower = img.copy()

# Create a Gaussian Pyramid
gaussian_pyr = [lower]
for i in range(3):
   lower = cv2.pyrDown(lower)
   gaussian_pyr.append(lower)

# Last level of Gaussian remains same in Laplacian
laplacian_top = gaussian_pyr[-1]

# Create a Laplacian Pyramid
laplacian_pyr = [laplacian_top]
for i in range(3,0,-1):
   size = (gaussian_pyr[i - 1].shape[1], gaussian_pyr[i - 1].shape[0])
   gaussian_expanded = cv2.pyrUp(gaussian_pyr[i], dstsize=size)
   laplacian = cv2.subtract(gaussian_pyr[i-1], gaussian_expanded)
   laplacian_pyr.append(laplacian)

# create three windows to display three layers of images
# cv2.namedWindow('Layer 1', cv2.WINDOW_AUTOSIZE)
# cv2.namedWindow('Layer 2', cv2.WINDOW_AUTOSIZE)
# cv2.namedWindow('Layer 3', cv2.WINDOW_AUTOSIZE)

# display all three layers
cv2.imwrite('Layer_1.png',laplacian_pyr[3])
cv2.imwrite('Layer_2.png',laplacian_pyr[2])
cv2.imwrite('Layer_3.png',laplacian_pyr[1])
