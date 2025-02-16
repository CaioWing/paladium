import click
import cv2  # Needed for image I/O
from video_processor import VideoProcessor
from image_processor import ImageProcessor


@click.group()
def cli():
    pass


@cli.command()
@click.option("--input", required=True, help="Path to the video")
@click.option("--output", default="output.csv", help="Path to the output CSV")
@click.option(
    "--video-output",
    default=None,
    help="Path to save the annotated output video (opcional)",
)
@click.option("--debug", is_flag=True, help="Ativa o modo de debug em live")
def video(input, output, video_output, debug):
    processor = VideoProcessor()
    processor.process_video(input, output, video_output, debug=debug)
    if video_output:
        print(f"Results saved in {output} and annotated video saved in {video_output}")
    else:
        print(f"Results saved in {output}")


@cli.command()
@click.option("--input", required=True, help="Path to the image")
def image(input):
    processor = ImageProcessor()
    frame = cv2.imread(input)
    results, annotated_image = processor.process_image(frame)
    for plate in results.values():
        print(
            f"Detected plate: {plate['vehicle'].plate.text} (Confidence: {plate['vehicle'].plate.ocr_confidence})"
        )


if __name__ == "__main__":
    cli()
