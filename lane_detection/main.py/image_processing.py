
import cv2
import os
import numpy as np  # NumPy is used for creating masks

image_folder = r"C:\Users\THINPAD\Desktop\pythonproject\lane_detection\images"
output_folder = r"C:\Users\THINPAD\Desktop\pythonproject\lane_detection\output"
os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(image_folder):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        img_path = os.path.join(image_folder, filename)
        img = cv2.imread(img_path)

        # Preprocessing
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5,5), 0)

        # Canny edge detection
        edges = cv2.Canny(blur, 50, 150)

        # --- NumPy mask for region of interest ---
        height, width = edges.shape
        mask = np.zeros_like(edges)
        polygon = np.array([[
            (0, height),
            (width, height),
            (width//2, height//2),
            (width//2, height//2)  # simple triangle, adjust if needed
        ]], np.int32)
        cv2.fillPoly(mask, polygon, 255)
        masked_edges = cv2.bitwise_and(edges, mask)

        # Hough Transform on masked edges
        lines = cv2.HoughLinesP(masked_edges, rho=1, theta=1*(3.14159/180),
                                threshold=50, minLineLength=50, maxLineGap=10)
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(img, (x1,y1), (x2,y2), (0,255,0), 2)

        # Show results
        cv2.imshow("Original with Lanes", img)
        cv2.imshow("Masked Edges", masked_edges)
        cv2.waitKey(0)

        # Save output
        cv2.imwrite(os.path.join(output_folder, filename), img)

cv2.destroyAllWindows()


#lane detection for the video
cap = cv2.VideoCapture("test_video")
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    # Apply same lane detection steps here
cap.release()
cv2.destroyAllWindows()




import cv2
import numpy as np

# Path to video file
video_path = r"C:\Users\THINPAD\Desktop\pythonproject\lane_detection\videos\test_video"
output_path = r"C:\Users\THINPAD\Desktop\pythonproject\lane_detection\output"

# Open the video
cap = cv2.VideoCapture(video_path)

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Create VideoWriter to save the output
out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale and blur
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)

    # Canny edge detection
    edges = cv2.Canny(blur, 50, 150)

    # Region of interest (mask)
    height, width = edges.shape
    mask = np.zeros_like(edges)
    polygon = np.array([[
        (0, height),
        (width, height),
        (width//2, height//2),
        (width//2, height//2)
    ]], np.int32)
    cv2.fillPoly(mask, polygon, 255)
    masked_edges = cv2.bitwise_and(edges, mask)

    # Hough line detection
    lines = cv2.HoughLinesP(masked_edges, rho=1, theta=1*(np.pi/180),
                            threshold=50, minLineLength=50, maxLineGap=10)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(frame, (x1, y1), (x2, y2), (0,255,0), 2)

    # Show the result
    cv2.imshow("Lane Detection Video", frame)

    # Write the frame to output video
    out.write(frame)

    # Press 'q' to quit early
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release everything
cap.release()
out.release()
cv2.destroyAllWindows()




