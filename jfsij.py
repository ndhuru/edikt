import cv2
import numpy as np

# Function to preprocess the image
def preprocess_image(frame):
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply GaussianBlur to reduce noise and improve edge detection
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Use Canny edge detector to find edges in the image
    edges = cv2.Canny(blurred, 50, 150)

    return edges

# Function to find circles in the image
def find_circles(edges):
    circles = cv2.HoughCircles(
        edges,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=20,
        param1=10,
        param2=35,
        minRadius=320,
        maxRadius=350
    )

    return circles

# Function to filter circles based on a specified radius
def filter_circles(circles, target_radius):
    filtered_circles = []

    if circles is not None:
        for circle in circles[0, :]:
            radius = circle[2]
            if target_radius - 10 <= radius <= target_radius + 10:  # Allow a range of 10 pixels around the target radius
                filtered_circles.append(circle)

    return np.array(filtered_circles)

# Specify the path to your video file
video_path = 'IMG_3827.MOV'

# Open a video capture object using the video file
cap = cv2.VideoCapture(video_path)

# Create windows for largest circle, edge detection, and all circles
cv2.namedWindow('Largest Circle', cv2.WINDOW_NORMAL)
cv2.namedWindow('Edge Detection', cv2.WINDOW_NORMAL)
cv2.namedWindow('All Circles', cv2.WINDOW_NORMAL)

# Specify the target radius for filtering circles in the edge detection window
target_radius_edge_detection = 225

while True:
    # Capture the frame from the video
    ret, frame = cap.read()

    if not ret:
        break

    # Preprocess the frame
    edges = preprocess_image(frame)

    # Find all circles in the image
    circles = find_circles(edges)

    # Draw only the largest circle if found and display in the first window
    if circles is not None:
        circles = np.uint16(np.around(circles))
        largest_circle = max(circles[0, :], key=lambda x: x[2])  # Find the circle with the largest radius
        largest_center = (largest_circle[0], largest_circle[1])
        largest_radius = largest_circle[2]

        # Draw the largest circle with thicker lines
        largest_frame = frame.copy()
        cv2.circle(largest_frame, largest_center, largest_radius, (0, 255, 0), 10)  # Increase thickness to 10
        cv2.circle(largest_frame, largest_center, 10, (0, 0, 255), -1)  # Increase thickness to 10
        cv2.imshow('Largest Circle', largest_frame)

    # Display the original edge detection result in the second window
    cv2.imshow('Edge Detection', edges)

    # Draw all circles on the frame if found and display in the third window
    if circles is not None:
        all_circles_frame = frame.copy()
        circles = np.uint16(np.around(circles))
        for circle in circles[0, :]:
            center = (circle[0], circle[1])
            radius = circle[2]

            # Draw all circles on the frame with thicker lines
            cv2.circle(all_circles_frame, center, radius, (0, 255, 0), 10)  # Increase thickness to 10
            cv2.circle(all_circles_frame, center, 10, (0, 0, 255), -1)  # Increase thickness to 10

        cv2.imshow('All Circles', all_circles_frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
