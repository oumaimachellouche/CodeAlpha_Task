
import cv2
import numpy as np
import os

# --- Paths ---
video_path = r"C:\Users\THINPAD\Desktop\pythonproject\lane_detection\videos\test_video.mp4"
output_folder = r"C:\Users\THINPAD\Desktop\pythonproject\lane_detection\output_videos.mp4"
os.makedirs(output_folder, exist_ok=True)
output_video_path = os.path.join(output_folder, "output_video.mp4")

# --- Open video ---
cap = cv2.VideoCapture(video_path)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# --- Video writer ---
out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

# --- Process frames ---
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Grayscale + blur
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)

    # Canny edges
    edges = cv2.Canny(blur, 50, 150)

    # Trapezoid mask (focus on road lanes)
    height, width = edges.shape
    mask = np.zeros_like(edges)
    polygon = np.array([[
        (int(0.1*width), height),          # bottom-left
        (int(0.9*width), height),          # bottom-right
        (int(0.55*width), int(0.6*height)),# top-right
        (int(0.45*width), int(0.6*height)) # top-left
    ]], np.int32)
    cv2.fillPoly(mask, polygon, 255)
    masked_edges = cv2.bitwise_and(edges, mask)

    # Hough lines
    lines = cv2.HoughLinesP(masked_edges, rho=1, theta=np.pi/180,
                            threshold=50, minLineLength=50, maxLineGap=20)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(frame, (x1, y1), (x2, y2), (0,255,0), 3)

    # Show and save
    cv2.imshow("Lane Detection Video", frame)
    out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
print("Video processing finished. Saved to:", output_video_path)
