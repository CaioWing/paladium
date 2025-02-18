# /paladium/video_processor.py

import cv2
import numpy as np
import time
from image_processor import ImageProcessor
from tqdm import tqdm
from utils.file_io import write_csv
from utils.ocr import read_license_plate
from utils.preprocessing import preprocess_license_plate
from utils.plate_data import LicensePlate
from utils.vehicle_data import Vehicle
import os


class Track:
    def __init__(self, bbox, track_id, last_frame=0):
        self.bbox = bbox  # [x1, y1, x2, y2]
        self.track_id = track_id
        self.lost_frames = 0
        self.plate_history = (
            []
        )  # Lista de tuplas: (ocr_confidence, LicensePlate, vehicle_bbox)
        self.last_frame = last_frame


class Tracker:
    def __init__(self, iou_threshold=0.3, max_lost=5):
        self.tracks = {}  # dict de track_id: Track
        self.next_id = 0
        self.iou_threshold = iou_threshold
        self.max_lost = max_lost

    def iou(self, boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
        return interArea / float(boxAArea + boxBArea - interArea)

    def update(self, detections, frame_idx):
        """
        Atualiza os tracks com as novas detecções.
        :param detections: lista de [x1, y1, x2, y2, score]
        :param frame_idx: número do frame atual
        :return: lista de tracks no formato [x1, y1, x2, y2, track_id]
        """
        assigned_detections = set()
        # Tenta associar cada track existente à detecção de maior IOU
        for tid, track in list(self.tracks.items()):
            best_iou = 0
            best_det_idx = -1
            best_det = None
            for i, det in enumerate(detections):
                if i in assigned_detections:
                    continue
                current_iou = self.iou(track.bbox, det[:4])
                if current_iou > best_iou:
                    best_iou = current_iou
                    best_det = det
                    best_det_idx = i
            if best_iou >= self.iou_threshold:
                track.bbox = best_det[:4]
                track.lost_frames = 0
                track.last_frame = frame_idx
                assigned_detections.add(best_det_idx)
            else:
                track.lost_frames += 1

        # Cria novos tracks para detecções não associadas
        for i, det in enumerate(detections):
            if i not in assigned_detections:
                self.tracks[self.next_id] = Track(
                    bbox=det[:4], track_id=self.next_id, last_frame=frame_idx
                )
                self.next_id += 1

        # Remove tracks que estão perdidas há muitos frames
        to_remove = [
            tid
            for tid, track in self.tracks.items()
            if track.lost_frames > self.max_lost
        ]
        for tid in to_remove:
            del self.tracks[tid]

        # Retorna uma lista de tracks para anotação: [x1, y1, x2, y2, track_id]
        track_list = [
            [track.bbox[0], track.bbox[1], track.bbox[2], track.bbox[3], track.track_id]
            for track in self.tracks.values()
        ]
        return track_list


class VideoProcessor:
    def __init__(self, batch_size=1):
        self.image_processor = ImageProcessor()
        self.tracker = Tracker(iou_threshold=0.6, max_lost=5)
        self.batch_size = batch_size

    def annotate_frame(self, frame, results, tracks):
        """
        Desenha caixas azuis para os veículos (com ID).
        """
        # Desenha caixas dos veículos
        for t in tracks:
            x1, y1, x2, y2, tid = t
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
            cv2.putText(
                frame,
                f"ID: {tid}",
                (int(x1), int(y1) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 0, 0),
                2,
            )
        return frame

    def _process_car_plates(self, frame, track_ids):
        """
        Para cada frame, detecta placas e realiza OCR para as detecções
        que estão contidas na caixa do veículo.
        Retorna um dicionário {track_id: [(ocr_conf, LicensePlate, vehicle_bbox), ...]}
        e os tempos de detecção.
        """
        frame_plate_inferences = {}
        start_plate = time.perf_counter()
        plates = self.image_processor.detect_license_plates(frame)
        plate_detection_time = (time.perf_counter() - start_plate) * 1000

        total_ocr_time = 0.0
        for plate in plates:
            p_x1, p_y1, p_x2, p_y2, p_score, _ = plate
            for car in track_ids:
                x1, y1, x2, y2, car_id = car
                if p_x1 >= x1 and p_y1 >= y1 and p_x2 <= x2 and p_y2 <= y2:
                    license_plate_crop = frame[
                        int(p_y1) : int(p_y2), int(p_x1) : int(p_x2)
                    ]
                    processed_plate = preprocess_license_plate(license_plate_crop)
                    ocr_start = time.perf_counter()
                    text, text_score = read_license_plate(processed_plate)
                    total_ocr_time += (time.perf_counter() - ocr_start) * 1000
                    if text:
                        lp = LicensePlate(
                            bbox=[p_x1, p_y1, p_x2, p_y2],
                            text=text,
                            detection_confidence=p_score,
                            ocr_confidence=text_score,
                        )
                        vehicle_bbox = [x1, y1, x2, y2, 0]
                        if car_id not in frame_plate_inferences:
                            frame_plate_inferences[car_id] = []
                        frame_plate_inferences[car_id].append(
                            (lp.ocr_confidence, lp, vehicle_bbox)
                        )
                    break  # Já associou a placa a este veículo
        return frame_plate_inferences, plate_detection_time, total_ocr_time

    def _annotate_debug_plate_detections(self, frame, frame_plate_inferences):
        """
        Em modo debug, desenha todas as caixas de placa (amarelo) e o texto de OCR.
        """
        for car_id, inferences in frame_plate_inferences.items():
            for inference in inferences:
                ocr_conf, lp, veh_bbox = inference
                x1, y1, x2, y2 = map(int, lp.bbox)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                cv2.putText(
                    frame,
                    lp.text,
                    (x1, y1 - 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2,
                )
        return frame

    def process_video(
        self, input_path, output_csv=None, output_video=None, frame_skip=10, debug=False
    ):
        """
        Processa o vídeo acumulando as inferências de placa para cada veículo.
        No final, gera uma saída CSV (se output_csv for informado) e retorna os resultados.
        Se debug estiver ativo, também retorna a última imagem anotada.
        """
        ALLOWED_VIDEO_FORMATS = [".mp4", ".avi", ".mkv", ".mov"]
        if not any(input_path.lower().endswith(ext) for ext in ALLOWED_VIDEO_FORMATS):
            print("Formato de vídeo não suportado.")
            return

        cap = cv2.VideoCapture(input_path)
        ret, first_frame = cap.read()
        if not ret:
            print("Erro ao ler o vídeo")
            return

        height, width, _ = first_frame.shape
        fps = cap.get(cv2.CAP_PROP_FPS)
        if output_video:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video_writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
        else:
            video_writer = None

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        pbar = tqdm(total=total_frames, desc="Processando vídeo")

        frames_buffer = [first_frame]
        frame_indices = [0]
        final_results = {}  # Dicionário com uma linha por veículo (chave: track_id)
        frame_counter = 0
        finished = False
        last_annotated = None  # Armazena o último frame anotado

        while not finished:
            while len(frames_buffer) < self.batch_size:
                ret, frame = cap.read()
                if not ret:
                    finished = True
                    break
                frame_counter += 1
                if frame_counter % frame_skip != 0:
                    continue
                frames_buffer.append(frame)
                frame_indices.append(frame_counter)
                pbar.update(frame_skip if frame_skip > 1 else 1)

            if len(frames_buffer) == 0:
                break

            for i, frame in enumerate(frames_buffer):
                current_frame_idx = frame_indices[i]
                # --- DETECÇÃO DE VEÍCULOS ---
                start_vehicle = time.perf_counter()
                vehicles = self.image_processor.detect_vehicles(frame)
                vehicle_detection_time = (time.perf_counter() - start_vehicle) * 1000

                # Atualiza o tracker com o frame atual
                tracks = self.tracker.update(np.array(vehicles), current_frame_idx)

                # --- DETECÇÃO DE PLACAS + OCR ---
                frame_plate_inferences, plate_detection_time, ocr_detection_time = (
                    self._process_car_plates(frame, tracks)
                )

                # Atualiza o histórico de OCR de cada track
                for tid, inferences in frame_plate_inferences.items():
                    if tid in self.tracker.tracks:
                        self.tracker.tracks[tid].plate_history.extend(inferences)

                # Monta dicionário para anotação (apenas se houver histórico de placa)
                annotation_dict = {}
                for t in tracks:
                    x1, y1, x2, y2, tid = t
                    if (
                        tid in self.tracker.tracks
                        and self.tracker.tracks[tid].plate_history
                    ):
                        best_entry = max(
                            self.tracker.tracks[tid].plate_history, key=lambda x: x[0]
                        )
                        best_plate = best_entry[1]
                        vehicle_obj = Vehicle(
                            bbox=[x1, y1, x2, y2, 0],
                            id=tid,
                            plate=best_plate,
                            frame_nmr=current_frame_idx,
                        )
                        annotation_dict[tid] = {"vehicle": vehicle_obj}

                annotated_frame = self.annotate_frame(
                    frame.copy(), annotation_dict, tracks
                )
                if debug:
                    annotated_frame = self._annotate_debug_plate_detections(
                        annotated_frame, frame_plate_inferences
                    )
                    debug_text = (
                        f"Veiculo: {vehicle_detection_time:.1f}ms | "
                        f"Placa: {plate_detection_time:.1f}ms | OCR: {ocr_detection_time:.1f}ms"
                    )
                    cv2.putText(
                        annotated_frame,
                        debug_text,
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2,
                    )
                    cv2.imshow("Debug", annotated_frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        finished = True
                        break

                if video_writer is not None:
                    video_writer.write(annotated_frame)

            frames_buffer = []
            frame_indices = []

        # Ao final, para cada track persistente, seleciona a melhor placa
        for tid, track in self.tracker.tracks.items():
            if track.plate_history:
                best_entry = max(track.plate_history, key=lambda x: x[0])
                best_plate = best_entry[1]
                vehicle_obj = Vehicle(
                    bbox=track.bbox + [0],
                    id=tid,
                    plate=best_plate,
                    frame_nmr=track.last_frame,
                )
                final_results[tid] = {"vehicle": vehicle_obj}

        cap.release()
        if video_writer is not None:
            video_writer.release()
        pbar.close()
        if debug:
            cv2.destroyAllWindows()

        # Imprime no terminal as placas únicas (uma por veículo)
        unique_plates = set()
        for tid, data in final_results.items():
            plate_text = data["vehicle"].plate.text if data["vehicle"].plate else ""
            if plate_text and plate_text not in unique_plates:
                print(f"Vehicle ID {tid}: Plate {plate_text}")
                unique_plates.add(plate_text)

        # Escreve o CSV se um caminho for informado
        if output_csv:
            write_csv(final_results, output_csv)

        # Se estiver em modo debug, retorna também a última imagem anotada
        if debug:
            return final_results, last_annotated
        else:
            return final_results
