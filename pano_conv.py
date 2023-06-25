import numpy as np
import cv2
import matplotlib.pyplot as plt
import time 

# calculate feature descriptors of an input img
def featureDesc(img, ipoints) : 
	im = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	h, w = im.shape
	
	features = []
	nipoints = []
	for point in ipoints :
		if point[0]+20 < w-1 and point[0]-20 >= 0 and point[1]+20 < h-1 and point[1]-20 >= 0 :
			crop = im[point[1]-20:point[1]+20, point[0]-20:point[0]+20]
			g_blur = cv2.GaussianBlur(crop, ksize=(5,5), sigmaX=0, sigmaY=0)
			res = cv2.resize(g_blur, None, fx=0.2, fy=0.2, interpolation = cv2.INTER_CUBIC)
			res = res.reshape((64,1))
			mean = np.mean(res)
			sd = np.std(res)
			res = (res-mean)/(sd+10e-7)
			features.append(res)
			nipoints.append(point)

	return features, nipoints

# Adaptive Non-Maximum Suppression
# performs corner detection using Shi-Tomasi Corner Detector & Good Features to Track
def ANMS(im, num) :

	flag = True
	im1 = im.copy()
	gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
	Nbest = 500
	Ncorners = 2*Nbest
	corners = cv2.goodFeaturesToTrack(gray,Ncorners,0.01,5)
	if type(corners) == type(None) : 
		flag = False
		return im, im, [], flag

	cor_num = len(corners)
	ipoints = []

	for i in range(cor_num) :
		xi, yi= int(corners[i][0][0]), int(corners[i][0][1])
		ipoints.append([xi, yi])
		# cv2.circle(im1, (xi, yi), 2, (0, 0, 255), -1)
		# cv2.imwrite(f"matching/anms{num}.png", im1)

	return im, im1, ipoints, flag


# calculates the sum of squared differences
def findSSD(feature1, feature2) :
	ssd = np.sum((feature1-feature2)**2, axis=0)
	return ssd

# resize two images to have the same dimensions
def equateSize(im1, im2):

	img1 = cropImage(im1.copy())
	img2 = cropImage(im2.copy())
	h1, w1 = img1.shape[:2]
	h2, w2 = img2.shape[:2]

	h = max(h1, h2)
	w = max(w1, w2) 

	img1 = cv2.resize(img1, (h,w))
	img2 = cv2.resize(img2, (h,w))

	return img1, img2

# # visualize the matches between keypoints in two input images
# def drawMatches(im1, keypoints1, im2, keypoints2) :

# 	img1, img2 = equateSize(im1.copy(), im2.copy())
# 	new_img = np.concatenate((img1, img2), axis=1)
# 	numkeypoints = len(keypoints1)

# 	r = 4
# 	thickness = 1
	
# 	for i in range(numkeypoints) :

# 		end1 = keypoints1[i]
# 		end2 = (keypoints2[i][0]+img1.shape[1], keypoints2[i][1])

# 		cv2.line(new_img, end1, end2, (0,255,255), thickness)
# 		cv2.circle(new_img, end1, r, (0,0,255), thickness)
# 		cv2.circle(new_img, end2, r, (0,255,0), thickness)

# 	return new_img


def featureMatch(im1, im2, num):
    keypoints1 = []
    keypoints2 = []
    matchRatio = 0.8

    anmsc1, corner1, ipoints1, flag1 = ANMS(im1, num)
    features1, nipoints1 = featureDesc(im1, ipoints1)

    anmsc2, corner2, ipoints2, flag2 = ANMS(im2, num + 1)
    features2, nipoints2 = featureDesc(im2, ipoints2)

    # Check if features were found in either image
    if not features1 or not features2:
        return keypoints1, keypoints2, False

    for i, feature1_tmp in enumerate(features1):
        difference = np.sum(np.abs(features2 - feature1_tmp), axis=1)
        dsort = np.sort(difference, axis=0)
        indexes = np.argsort(difference, axis=0)

        # Calculate the match ratio for the best and second-best matches
        mratio = dsort[0][0] / (dsort[1][0] + 1e-3)

        if mratio < matchRatio:
            keypoints1.append(nipoints1[i])
            keypoints2.append(nipoints2[indexes[0][0]])

    return keypoints1, keypoints2, (flag1 or flag2)


# Performs RANSAC algorithm to reject outliers 
# and estimate the homography matrix between matched keypoints in two images.
def my_ransac(keypoints1, keypoints2, im1, im2, num):
    sample_size = len(keypoints1)
    max_inliers = 0
    num_iterations = 2000
    thresh = 5000
    n = 4
    flag = True
    
    homo_mat = np.zeros((3, 3))
    final_inliers = []
    
    # RANSAC iterations
    for interation in range(num_iterations):
        indices = np.random.choice(sample_size, n, replace=False)
        pi = np.array([keypoints1[idx] for idx in indices])
        pii = np.array([keypoints2[idx] for idx in indices])
        
        # Estimate homography matrix using the selected keypoints
        H, status = cv2.findHomography(pi, pii)
        inliers = []
        
        if H is not None:
            # Calculate the number of inliers for the estimated homography matrix
            for kp1, kp2 in zip(keypoints1, keypoints2):
                kpm2 = np.array([kp2[0], kp2[1], 1]).T
                kpm1 = np.array([kp1[0], kp1[1], 1]).T
                ssd = findSSD(kpm2, np.dot(H, kpm1))
                
                # If the sum of squared differences is below the threshold, consider it as an inlier
                if ssd < thresh:
                    inliers.append([kp1, kp2])
            
            # Update the maximum number of inliers and the corresponding homography matrix
            if len(inliers) > max_inliers:
                max_inliers = len(inliers)
                homo_mat = H
                final_inliers = inliers
    
    # Extract keypoints from final inliers
    kpts1 = [item[0] for item in final_inliers]
    kpts2 = [item[1] for item in final_inliers]
    
    # Check if there are enough keypoints for reliable estimation
    if len(kpts1) < 40 or len(kpts2) < 40:
        return [], homo_mat, False
    
    homo_mat, mask = cv2.findHomography(np.array(kpts1), np.array(kpts2))
    
    return final_inliers, homo_mat, flag


# overlay img1 onto img2 
# copy non-zero pixel values from wimg1 to the corresponding positions in wimg2.
def stitchImage(img1, img2) :
	for i in range(img2.shape[0]) :
		for j in range(img2.shape[1]) :
			if img1[i, j].any() > 0 :
				img2[i, j] =img1[i,j]

	return img2
	
# perform a series of transformations and stitching on the images in img_list 
# and returns the final stitched image

def transform(img_list, counter) :

	img1 = img_list[0]

	for i in range(1,len(img_list)) :
		img2 = img_list[i]
		tmp1 = time.time()
		keypoints1, keypoints2, flag = featureMatch(img1, img2, i+counter)
		tmp2 = time.time()
		time_featureMatch = tmp2 - tmp1
		# print ("time of featureMatch(sec)",time_featureMatch)
		print(f'Keypoints after matching : {len(keypoints1)}')

		if not flag : 
			print(f'Found few features or not a good match')
			continue
			
		if len(keypoints1) < 40 or len(keypoints2) < 40 : 
			print(f'Found less than 40 features')
			continue
		tmp1 = time.time()
		finalInliers, Hbest, flag = my_ransac(keypoints1, keypoints2, img1, img2, i+counter)
		tmp2 = time.time()
		time_my_ransac = tmp2 - tmp1
		# print ("time of my_ransac(sec)",time_my_ransac)
		if not flag or len(finalInliers) < 40 :
			continue
	
		
		h1, w1 = img1.shape[0], img1.shape[1]
		h2, w2 = img2.shape[0], img2.shape[1]

		points1 = np.array([[0,0], [w1,0], [0,h1],[w1,h1]])
		points2 = np.array([[0,0], [w2,0], [0,h2],[w2,h2]])

		points1 = points1.reshape(-1,1,2).astype(np.float32)
		if type(Hbest) == type(None) : 
			print('None')
			continue

		wpoints1 = cv2.perspectiveTransform(points1,  Hbest).reshape(-1,2)
		
		rpoints = np.concatenate((wpoints1, points2), axis=0)

		xmin, ymin = int(np.min(rpoints, axis=0)[0]), int(np.min(rpoints, axis=0)[1])
		xmax, ymax = int(np.max(rpoints, axis=0)[0]), int(np.max(rpoints, axis=0)[1])
	
		Htrans = np.array([[1, 0, -xmin], [0, 1, -ymin], [0, 0, 1]]).astype(float)

		Hbest = Htrans@Hbest

		height = ymax-ymin
		width = xmax-xmin
		size = (height,width)
		
		wimg1 = cv2.warpPerspective(img1, Hbest, (2000,2000)) 
		wimg2 = cv2.warpPerspective(img2, Htrans, (2000,2000)) 

		mask1 = cv2.threshold(wimg1, 0, 255, cv2.THRESH_BINARY)[1]
		kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
		mask1 = cv2.morphologyEx(mask1, cv2.MORPH_ERODE, kernel1)
		wimg1[mask1==0] = 0

		mask2 = cv2.threshold(wimg2, 0, 255, cv2.THRESH_BINARY)[1]
		kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
		mask2 = cv2.morphologyEx(mask2, cv2.MORPH_ERODE, kernel2)
		wimg2[mask2==0] = 0
		
		img1 = stitchImage(wimg1, wimg2)


	return img1

# perform cropping to remove black borders or regions of zero intensity pixels
def cropImage(image) :
	h, w = image.shape[:2]
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	first_passc = True
	first_passr = True
	pixelscol = np.sum(gray, axis=0).tolist()
	nresult = image
	for index, value in enumerate(pixelscol):
		if value == 0:
			continue
		else:
			ROI = image[0:h, index:index+1]
			if first_passc:
				result = image[0:h, index+1:index+2]
				first_passc = False
				continue
			result = np.concatenate((result, ROI), axis=1)

	
	gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
	pixelsrow = np.sum(gray, axis=1).tolist()
	h, w = result.shape[:2]
	for index, value in enumerate(pixelsrow):
		if value == 0:
			continue
		else:
			ROI = result[index:index+1, 0:w]
			if first_passr:
				result1 = result[index+1:index+2, 0:w]
				first_passr = False
				continue
			result1 = np.concatenate((result1, ROI), axis=0)

	return result1
	


def converter(img_list):

	global status
	all_imgs = img_list[:]	
	# my functions
	# 	
	tmp1 = time.time()
	img = transform(all_imgs, 0)
	tmp2 = time.time()
	time_transform = tmp2 - tmp1

	tmp1 = time.time()
	img = cropImage(img)
	tmp2 = time.time()
	time_cropImage = tmp2 - tmp1

	# # cv solution	
	# stitcher=cv2.Stitcher.create()
	# status,img=stitcher.stitch(all_imgs)

	status = 1
	# print ("time of transform(sec)",time_transform)
	# print("time of cropImage(sec)",time_cropImage)
	return status, img

	
	

 
