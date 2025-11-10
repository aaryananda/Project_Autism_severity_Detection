# Install yt-dlp if not already installed
pip install yt-dlp

# Import os just for organizing downloads (optional)
import os

# Create a folder to save videos
os.makedirs("youtube_videos", exist_ok=True)

# Example YouTube video URL
video_url = "https://www.youtube.com/watch?v=YOUR_VIDEO_ID"

# Use yt-dlp to download video in best quality
!yt-dlp -f best -o "youtube_videos/%(title)s.%(ext)s" {video_url}
