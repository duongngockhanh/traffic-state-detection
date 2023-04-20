from moviepy.video.io.VideoFileClip import VideoFileClip

# Set up input and output file names
input_file = "a5_non_hd_front_congestion_01.mp4"
output_file = "a0500000300.mp4"

# Set start and end times in seconds
start_time = 0
end_time = 10

# Open input video file and extract subclip
clip = VideoFileClip(input_file).subclip(start_time, end_time)

# Write subclip to output file
clip.write_videofile(output_file)

# Close input and output files
clip.reader.close()
clip.audio.reader.close_proc()
