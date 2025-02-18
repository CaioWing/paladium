import click
import cv2
import yt_dlp
import os
import tempfile
import warnings
import sys

warnings.filterwarnings("ignore", category=FutureWarning)


def is_youtube_url(url):
    """Verifica se a entrada é uma URL do YouTube."""
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
    Faz o download do vídeo do YouTube para um arquivo temporário e retorna o caminho.
    Retorna None se o download falhar.
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
    """Grupo de comandos CLI."""
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
    from video_processor import VideoProcessor

    """Processa um arquivo de vídeo ou URL do YouTube para detecção de placas."""
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
    from image_processor import ImageProcessor

    """Processa uma única imagem para detecção de placas."""
    processor = ImageProcessor()
    frame = cv2.imread(input)
    results, annotated_image = processor.process_image(frame)
    for plate in results.values():
        print(
            f"Detected plate: {plate['vehicle'].plate.text} (Confidence: {plate['vehicle'].plate.ocr_confidence})"
        )


@cli.command()
@click.option(
    "--input", required=True, help="Path to folder containing images or videos"
)
@click.option(
    "--output", default="output.csv", help="Path to the aggregated output CSV file"
)
@click.option(
    "--frame-skip", default=2, help="Number of frames to skip for video processing"
)
@click.option(
    "--debug", is_flag=True, help="Enable live debug mode for video processing"
)
def folder(input, output, frame_skip, debug):
    """
    Processa todos os vídeos ou imagens em uma pasta e suas subpastas para detecção de placas.
    Agrega os resultados em um único arquivo CSV, incluindo o caminho relativo do arquivo.
    No modo debug:
      - Para imagens: exibe a imagem anotada e aguarda a tecla 'c' para continuar.
      - Para vídeos: o debug é exibido ao vivo dentro do processamento do vídeo.
    """
    import csv
    import glob
    from image_processor import ImageProcessor
    from video_processor import VideoProcessor

    # Define extensões de arquivos suportadas.
    video_extensions = (".mp4", ".avi", ".mkv", ".mov")
    image_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")

    aggregated_rows = (
        []
    )  # Cada linha: [file_path, source_file, frame_nmr, car_id, car_bbox, license_plate_bbox, license_plate_bbox_score, license_number, license_number_score]

    # Glob para buscar recursivamente todos os arquivos (incluindo subdiretórios)
    pattern = os.path.join(input, "**", "*")
    files = glob.glob(pattern, recursive=True)

    # Instancia os processadores uma única vez para todos os arquivos
    ip = ImageProcessor()
    vp = VideoProcessor()

    for file_path in files:
        if os.path.isfile(file_path):
            # Fecha todas as janelas abertas para evitar "travamento" entre arquivos
            cv2.destroyAllWindows()
            ext = os.path.splitext(file_path)[1].lower()
            relative_path = os.path.relpath(file_path, input)
            if ext in video_extensions:
                click.echo(f"Processing video: {relative_path}")
                # Para vídeos, o método process_video já exibe o debug ao vivo
                final_results = vp.process_video(
                    file_path,
                    output_csv=None,
                    output_video=None,
                    frame_skip=frame_skip,
                    debug=debug,
                )
                # Para vídeos, não aguardamos key press, deixando o debug ativo durante o processamento.
                for track_id, data in final_results.items():
                    if "vehicle" in data and data["vehicle"].plate is not None:
                        vehicle = data["vehicle"]
                        plate = vehicle.plate
                        car_bbox = vehicle.bbox[:4]
                        license_plate_bbox = plate.bbox
                        aggregated_rows.append(
                            [
                                relative_path,
                                os.path.basename(file_path),
                                vehicle.frame_nmr,
                                track_id,
                                f"[{' '.join(str(x) for x in car_bbox)}]",
                                f"[{' '.join(str(x) for x in license_plate_bbox)}]",
                                plate.detection_confidence,
                                plate.text,
                                plate.ocr_confidence,
                            ]
                        )
            elif ext in image_extensions:
                click.echo(f"Processing image: {relative_path}")
                frame = cv2.imread(file_path)
                if frame is None:
                    click.echo(f"Warning: Could not read image {relative_path}")
                    continue
                results, annotated_image = ip.process_image(frame)
                if debug:
                    cv2.imshow("Debug - Annotated Image", annotated_image)
                    key = cv2.waitKey(0)
                    if key == ord("c"):
                        cv2.destroyWindow("Debug - Annotated Image")
                for track_id, data in results.items():
                    if "vehicle" in data and data["vehicle"].plate is not None:
                        vehicle = data["vehicle"]
                        plate = vehicle.plate
                        car_bbox = vehicle.bbox[:4]
                        license_plate_bbox = plate.bbox
                        aggregated_rows.append(
                            [
                                relative_path,
                                os.path.basename(file_path),
                                vehicle.frame_nmr,
                                track_id,
                                f"[{' '.join(str(x) for x in car_bbox)}]",
                                f"[{' '.join(str(x) for x in license_plate_bbox)}]",
                                plate.detection_confidence,
                                plate.text,
                                plate.ocr_confidence,
                            ]
                        )
            else:
                click.echo(f"Skipping unsupported file: {relative_path}")

    # Escreve os resultados agregados no arquivo CSV, incluindo a informação do caminho do arquivo.
    with open(output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "file_path",
                "source_file",
                "frame_nmr",
                "car_id",
                "car_bbox",
                "license_plate_bbox",
                "license_plate_bbox_score",
                "license_number",
                "license_number_score",
            ]
        )
        for row in aggregated_rows:
            writer.writerow(row)

    click.echo(f"Aggregated results saved to {output}")


if __name__ == "__main__":
    cli()
