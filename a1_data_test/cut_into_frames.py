import cv2

# Open the video file
video = cv2.VideoCapture('a0300000300.mp4')

# Initialize frame count and set the interval to extract frames
frame_count = 0
interval = 5 # Extract every 10th frame

# Loop through the frames
while True:
    # Read the next frame from the video
    ret, frame = video.read()

    # Check if the frame was successfully read
    if not ret:
        break

    # Increment the frame count
    frame_count += 1

    # Check if it's time to extract a frame
    if frame_count % interval == 0:
        # Generate a filename for the frame
        filename = f'a0300000300_{frame_count}.jpg'

        # Save the frame as an image file
        cv2.imwrite(filename, frame)

# Release the video file
video.release()
