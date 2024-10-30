import numpy as np
import cv2
import glob

# Define the size of the chessboard 
board_size = (11, 7)
objp = np.zeros((board_size[0] * board_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2)

# Arrays to store object points and image points
objpoints = []  # 3D points
imgpoints = []  # 2D points

images = glob.glob('images/*.png')

for fname in images:
    print(f"Processing file: {fname}")
    img = cv2.imread(fname)
    if img is None:
        print(f"Failed to load image:< {fname}")
        continue

    img = cv2.resize(img, (600, 400))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    cv2.imshow('Gray Image', gray)
    cv2.waitKey(500)

    ret, corners = cv2.findChessboardCorners(
        gray, board_size,
        cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FILTER_QUADS
    )

    if ret:
        print("Corners found!:>")
        objpoints.append(objp)
        imgpoints.append(corners)
        cv2.drawChessboardCorners(img, board_size, corners, ret)
        cv2.imshow('Detected Corners', img)
        cv2.waitKey(500)
    else:
        print(f"Corners not found in image: {fname}")

cv2.destroyAllWindows()

# Camera calibration
if len(objpoints) > 0:
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    print("Camera Matrix:\n", mtx)
    print("Distortion Coefficients:\n", dist)
else:
    print("No corners were detected .")

# Distortion correction
image_path = 'images/Im_L_6.png'
img = cv2.imread(image_path)
if img is None:
    print(f"Image not found or unable to load: {image_path}")
    exit()

h, w = img.shape[:2]

# Camera matrix and distortion coefficients 
mtx = np.array([[743.85093896, 0.0, 281.78630922], 
                 [0.0, 278.89236412, 202.43415084], 
                 [0.0, 0.0, 1.0]])
dist = np.array([0.15779989, -0.4753176, 0.00152599, 0.0009213, 0.57061802])

# Get the optimal new camera matrix
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

# Undistort the image
dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

if dst is None:
    print("Undistortion failed")
    exit()
else:
    print(f"Dimensions of dst: {dst.shape}")
    cv2.imshow('Undistorted Image', dst)  
    cv2.waitKey(0) 

# Print ROI values
print(f"ROI: {roi}")

# Ensure valid cropping
if roi == (0, 0, 0, 0):
    print("Invalid ROI, skipping cropping")
else:
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]

    if dst.size == 0:
        print("Crop operation resulted in an empty image")
        exit()

# Save the result
cv2.imwrite('calibrated_result.png', dst)

# Calculate and print reprojection error
mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
    mean_error += error

print("Total reprojection error: ", mean_error / len(objpoints))
