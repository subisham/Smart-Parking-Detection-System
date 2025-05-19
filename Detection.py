import cv2
import cv2.aruco as aruco

image_path = "aruco_marker_42_border.png"

# Load the image in grayscale (better for marker detection)
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
if img is None:
    print(f"Error: Could not load image from {image_path}")
    exit()
else:
    print(f"Image loaded successfully with shape {img.shape}")

# Get the dictionary and detection parameters
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
parameters = aruco.DetectorParameters()

# Detect markers in the grayscale image
corners, ids, rejected = aruco.detectMarkers(img, aruco_dict, parameters=parameters)

if ids is not None:
    print("Detected marker IDs:", ids.flatten())
    # Convert grayscale to BGR so we can draw colored markers
    img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    aruco.drawDetectedMarkers(img_color, corners, ids)
    cv2.imshow("Detected ArUco Markers", img_color)
else:
    print("No markers detected.")
    # Just show the grayscale image
    cv2.imshow("Detected ArUco Markers", img)

cv2.waitKey(0)
cv2.destroyAllWindows()