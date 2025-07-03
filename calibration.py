import numpy as np
import cv2
import os

# Loads/captures the feed
cap = cv2.VideoCapture(1) # 0 is integrated camera and 1 is external camera(webcam)
count = 0 # Keeps count of images taken
save_dir = "imag_webcam" # Directory to save images
calib_dir = "calib_webcam" # Directory to save the calibrated images
os.makedirs(save_dir, exist_ok=True) # Makes the directory for the raw images(no patterns show)
os.makedirs(calib_dir, exist_ok=True) # Makes the directory for the calibrated images

print("Press 's' to save image, 'q' to quit.")

while True: # Loop to capture the images
    ret, frame = cap.read()
    if not ret:
        break
    
    cv2.imshow("Live feed", cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))  # Show grayscale live feed
    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'): # Save images
        filename = os.path.join(save_dir, f"image_{count}.jpg")
        cv2.imwrite(filename, frame)
        print(f"Saved {os.path.basename(filename)}")
        count += 1
    elif key == ord('q'): # Quits the feed
        break

cap.release() # stops the video feed
cv2.destroyAllWindows() # makes sure to close all windows

# Here the Calibration process starts
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

chessboard_size = (8, 5)  # The inner corners of the chessboard
square_size_cm = 3.0  # Each square of the chessboard is 3 cm

objp = np.zeros((chessboard_size[1]*chessboard_size[0], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
objp *= square_size_cm  # Converts object points to real-world units (cm)

objpoints = []  # 3D points in real world
imgpoints = []  # 2D points in image

image_files = [os.path.join(save_dir, f) for f in os.listdir(save_dir) if f.endswith('.jpg')] # Uses the images taken from the camera

for idx, fname in enumerate(image_files): # Then loops through the images taken
    img = cv2.imread(fname) # Read the images
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # converts the images to grayscale

    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None) # Looks for the chessboard corners in the image

    if ret: # If the corners are found, it proceeds to refine the corners and save the points
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria) # Redefines the corners found further
        imgpoints.append(corners2)
        
        # Convert grayscale back to BGR to draw colored corners
        gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        cv2.drawChessboardCorners(gray_bgr, chessboard_size, corners2, ret) # Draws the pattern on the image
        cv2.imshow('Chessboard Corners', gray_bgr) # Shows the image with the corners drawn
        cv2.waitKey(500) # Wait for 500ms to see each image after calibrating

        # Save the grayscale image with the patterns drawn
        overlay_path = os.path.join(calib_dir, f"calibrated_overlay_webcam_{idx}.jpg")
        cv2.imwrite(overlay_path, gray_bgr)
        print(f"Saved calibrated overlay: {overlay_path}") # tells where the images are saved

cv2.destroyAllWindows() # closes all windows

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None) # Calibrates the camera using the points found in the images
tvecs_cm = np.asarray(tvecs, dtype=np.float64)  # These are now in centimeters (due to scaled objp)

# Displays the results of the calibration
print(f"\nReprojection Error (RMS): {ret}")
print("Camera Matrix:\n", mtx)
print("Distortion Coefficients:\n", dist)

# Load one image after calibration to undistort
img = cv2.imread(os.path.join(calib_dir, 'calibrated_overlay_webcam_3.jpg')) # Load the saved images after calibration to undistort
h,  w = img.shape[:2] # Get the height and width of the image
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h)) # Get the new optimized camera matrix and ROI, used to undistort the feed

print("Optimized Camera Matrix is:\n", newcameramtx)
print("ROI is:\n", roi) # Prints the ROI of the image

# Show estimated depth from camera to board in centimeters
print("\nEstimated distances to camera (Z-axis, cm):")
for i, tvec in enumerate(tvecs_cm):
    print(f"[{i}] Z = {tvec[2][0]:.2f} cm")

# Saves the results of the calibration in an npz file
np.savez(
    "optimized_camera_calib_webcam.npz",
    cameraMatrix      = mtx,
    distCoeffs        = dist,
    reprojectionError = ret,
    optCameraMatrix   = newcameramtx,
    roi               = np.asarray(roi),
    rvecs             = np.asarray(rvecs, dtype=np.float64),
    tvecs             = tvecs_cm,  # Already in cm
    square_size_cm    = square_size_cm
)
