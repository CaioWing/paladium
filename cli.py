import click
import cv2
from video_processor import VideoProcessor
from image_processor import ImageProcessor
import yt_dlp
import os
import tempfile
import warnings
import sys

warnings.filterwarnings("ignore", category=FutureWarning)


def is_youtube_url(url):
    """Check if the input is a YouTube URL."""
    return any(
        domain in url.lower()
        for domain in ["youtube.com/", "youtu.be/", "youtube.com/shorts/"]
    )


class DownloadLogger:
    def debug(self, msg):
        if msg.startswith("[debug] "):
            pass
        else:
            self.info(msg)

    def info(self, msg):
        if msg.startswith("[download] "):
            print(msg, file=sys.stderr)

    def warning(self, msg):
        print(f"Warning: {msg}", file=sys.stderr)

    def error(self, msg):
        print(f"Error: {msg}", file=sys.stderr)


def download_youtube_video(url):
    """
    Download YouTube video to a temporary file and return the path.
    Returns None if download fails.
    """
    try:
        # Create a temporary directory to handle the download
        temp_dir = tempfile.mkdtemp()
        temp_path = os.path.join(temp_dir, "video.mp4")

        ydl_opts = {
            "format": "best[ext=mp4][height<=1080]/best[ext=mp4]/best",  # Prefer MP4 up to 1080p
            "outtmpl": temp_path,
            "logger": DownloadLogger(),
            "progress_hooks": [
                lambda d: (
                    print(f"Downloading: {d['_percent_str']}", end="\r")
                    if d["status"] == "downloading"
                    else None
                )
            ],
            "merge_output_format": "mp4",
            "quiet": False,
            "no_warnings": False,
            "extract_flat": False,
            "writeinfojson": True,
            "retries": 3,
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # First, get video info
            try:
                print("Extracting video information...")
                info = ydl.extract_info(url, download=False)
                print(f"Found video: {info.get('title', 'Unknown title')}")
                print(f"Duration: {info.get('duration', 'Unknown')} seconds")
                print(f"Available quality: {info.get('format', 'Unknown')}")
            except Exception as e:
                print(f"Error extracting video info: {str(e)}")
                return None

            # Then download
            print("\nStarting download...")
            ydl.download([url])
            print("\nDownload completed. Verifying file...")

        # Verify the downloaded file
        if not os.path.exists(temp_path):
            print(f"Error: File not found at {temp_path}")
            return None

        if os.path.getsize(temp_path) == 0:
            print("Error: Downloaded file is empty")
            return None

        # Verify the file can be opened with OpenCV
        print("Verifying video file...")
        cap = cv2.VideoCapture(temp_path)
        if not cap.isOpened():
            print("Error: Downloaded video cannot be opened with OpenCV")
            cap.release()
            return None

        # Get video information
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"Video verified successfully:")
        print(f"Resolution: {width}x{height}")
        print(f"FPS: {fps}")
        print(f"Frame count: {frame_count}")

        cap.release()
        return temp_path

    except Exception as e:
        print(f"Error during video download/conversion: {str(e)}")
        if "temp_path" in locals() and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except:
                pass
        return None
    finally:
        # Cleanup any JSON files that yt-dlp might have created
        try:
            json_file = temp_path + ".info.json"
            if os.path.exists(json_file):
                os.unlink(json_file)
        except:
            pass


@click.group()
def cli():
    pass


@cli.command()
@click.option("--input", required=True, help="Path to video file or YouTube URL")
@click.option("--output", default="output.csv", help="Path to the output CSV")
@click.option("--frame-skip", default=2, help="Number of frames to skip each iteration")
@click.option(
    "--video-output",
    default=None,
    help="Path to save the annotated output video (optional)",
)
@click.option("--debug", is_flag=True, help="Enable live debug mode")
def video(input, output, video_output, frame_skip, debug):
    """Process a video file or YouTube URL for license plate detection."""
    video_path = input
    temp_path = None

    try:
        if is_youtube_url(input):
            temp_path = download_youtube_video(input)
            if not temp_path:
                print("Failed to download/convert YouTube video")
                return
            video_path = temp_path

        # Test video file before processing
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Cannot open video file: {video_path}")
            return
        cap.release()

        processor = VideoProcessor()
        processor.process_video(
            video_path, output, video_output, debug=debug, frame_skip=frame_skip
        )

        if video_output:
            print(
                f"Results saved in {output} and annotated video saved in {video_output}"
            )
        else:
            print(f"Results saved in {output}")

    finally:
        # Clean up temporary file if it was created
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
                os.rmdir(os.path.dirname(temp_path))
            except Exception as e:
                print(f"Warning: Could not delete temporary files: {str(e)}")


@cli.command()
@click.option("--input", required=True, help="Path to the image")
def image(input):
    """Process a single image for license plate detection."""
    processor = ImageProcessor()
    frame = cv2.imread(input)
    results, annotated_image = processor.process_image(frame)
    for plate in results.values():
        print(
            f"Detected plate: {plate['vehicle'].plate.text} (Confidence: {plate['vehicle'].plate.ocr_confidence})"
        )


if __name__ == "__main__":
    cli()
