
import numpy as np
import cv2


# input an image as a numpy array
# return image in same format
# where pixel values correspond to distance (pixels) to nearest obstacle
# obstacles are zeros
def distance_transform(image):


	image_reformat = np.float32(image)
	#print('reformat', image_reformat)

	# Threshold the image to convert a binary image
	ret, thresh = cv2.threshold(image_reformat, 50, 1, cv2.THRESH_BINARY_INV)

	#print('thresh',thresh)

	# Determine the distance transform.
	dist = cv2.distanceTransform(np.uint8(thresh), cv2.DIST_L2, 0)

	#print('dist',dist[10:20,10:20])
	  
	# Make the distance transform normal.
	#dist_output = cv2.normalize(dist, None, 0, 1.0, cv2.NORM_MINMAX)

	#print('dist',dist)
	  
	# Display the distance transform
	#cv2.imshow('Distance Transform', np.uint8(dist))
	#cv2.waitKey(0)

	return dist