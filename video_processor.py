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


class SimpleTracker:
    def __init__(self, iou_threshold=0.3):
        self.next_id = 0
        self.tracks = []  # Cada track: [x1, y1, x2, y2, id]
        self.iou_threshold = iou_threshold

    def iou(self, boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
        return interArea / float(boxAArea + boxBArea - interArea)

    def update(self, detections):
        """
        Atualiza os tracks com as novas detecções.
        detections: array numpy com shape (N, 5) onde cada linha é [x1, y1, x2, y2, score]
        """
        updated_tracks = []
        if len(self.tracks) == 0:
            for det in detections:
                x1, y1, x2, y2, score = det
                updated_tracks.append([x1, y1, x2, y2, self.next_id])
                self.next_id += 1
        else:
            assigned = [False] * len(detections)
            new_tracks = []
            for track in self.tracks:
                best_iou = 0
                best_det_idx = -1
                for idx, det in enumerate(detections):
                    if not assigned[idx]:
                        x1, y1, x2, y2, score = det
                        current_iou = self.iou(track[:4], [x1, y1, x2, y2])
                        if current_iou > best_iou:
                            best_iou = current_iou
                            best_det_idx = idx
                if best_iou >= self.iou_threshold and best_det_idx != -1:
                    det = detections[best_det_idx]
                    new_tracks.append([det[0], det[1], det[2], det[3], track[4]])
                    assigned[best_det_idx] = True
            for idx, det in enumerate(detections):
                if not assigned[idx]:
                    new_tracks.append([det[0], det[1], det[2], det[3], self.next_id])
                    self.next_id += 1
            updated_tracks = new_tracks

        self.tracks = updated_tracks
        return np.array(updated_tracks)


class VideoProcessor:
    def __init__(self, batch_size=1):
        self.image_processor = ImageProcessor()
        self.tracker = SimpleTracker()
        self.batch_size = batch_size
        # Histórico de inferências por veículo: {track_id: [(ocr_confidence, LicensePlate, vehicle_bbox), ...]}
        self.track_plate_history = {}
        self.active_tracks = set()  # IDs dos veículos atualmente visíveis
        self.final_results = {}  # resultados finais para escrita no CSV

    def annotate_frame(self, frame, results, tracks):
        """Desenha as caixinhas dos veículos (azul) e placas (a partir da otimização final – verde)."""
        # Desenha as caixas dos veículos (azul) com ID
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
        # Desenha as caixas de placas otimizadas (verde) e o texto da placa
        for detection in results.values():
            vehicle_obj = detection["vehicle"]
            if vehicle_obj.plate:
                plate = vehicle_obj.plate
                x1, y1, x2, y2 = map(int, plate.bbox)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    plate.text,
                    (x1, max(y1 - 10, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 0),
                    2,
                )
        return frame

    def _process_car_plates(self, frame, track_ids):
        """
        Para cada frame:
          1. Detecta todas as placas na imagem completa (mede o tempo de inferência do detector).
          2. Para cada placa, se estiver contida na caixa de um veículo, realiza OCR (mede o tempo do OCR).
          3. Retorna:
             - frame_plate_inferences: dicionário {track_id: [(ocr_conf, LicensePlate, vehicle_bbox), ...]}
             - plate_detection_time (ms)
             - ocr_detection_time (ms)
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
                # Se a placa estiver contida na caixa do veículo
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
                        vehicle_bbox = [x1, y1, x2, y2, 0]  # bbox do veículo
                        if car_id not in frame_plate_inferences:
                            frame_plate_inferences[car_id] = []
                        frame_plate_inferences[car_id].append(
                            (lp.ocr_confidence, lp, vehicle_bbox)
                        )
                    break  # já associou a placa a este veículo
        return frame_plate_inferences, plate_detection_time, total_ocr_time

    def _annotate_debug_plate_detections(self, frame, frame_plate_inferences):
        """
        Em debug, desenha todas as caixas de placas e os textos de OCR (cor amarelo)
        conforme são calculados em cada frame.
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
        self, input_path, output_csv, output_video=None, frame_skip=1, debug=False
    ):
        """
        Processa o vídeo, acumulando as inferências de placas para cada veículo.
        A otimização (seleção do melhor OCR/placa) é feita apenas para o resultado final.
        Se debug=True, além das anotações finais (em verde), são exibidas todas as
        detecções de placas e OCR (em amarelo) em cada frame.
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
        final_frame_results = {}  # resultados finais para CSV
        frame_counter = 0
        finished = False

        while not finished:
            # Preenche o buffer até batch_size
            while len(frames_buffer) < self.batch_size:
                ret, frame = cap.read()
                if not ret:
                    finished = True
                    break
                frame_counter += 1
                pbar.update(1)
                if frame_counter % frame_skip != 0:
                    continue
                frames_buffer.append(frame)
                frame_indices.append(frame_counter)

            if len(frames_buffer) == 0:
                break

            # Processa cada frame no batch
            for i, frame in enumerate(frames_buffer):
                # --- DETECÇÃO DE VEÍCULOS ---
                start_vehicle = time.perf_counter()
                vehicles = self.image_processor.detect_vehicles(frame)
                vehicle_detection_time = (time.perf_counter() - start_vehicle) * 1000

                tracks = self.tracker.update(np.array(vehicles))
                current_tracks = set([t[4] for t in tracks])

                # --- DETECÇÃO DE PLACAS + OCR ---
                frame_plate_inferences, plate_detection_time, ocr_detection_time = (
                    self._process_car_plates(frame, tracks)
                )

                # Acumula as inferências para cada veículo
                for tid, inferences in frame_plate_inferences.items():
                    if tid not in self.track_plate_history:
                        self.track_plate_history[tid] = []
                    self.track_plate_history[tid].extend(inferences)

                # Otimiza a inferência final: seleciona a melhor detecção acumulada para cada track
                annotation_dict = {}
                for t in tracks:
                    x1, y1, x2, y2, tid = t
                    if (
                        tid in self.track_plate_history
                        and self.track_plate_history[tid]
                    ):
                        best_entry = max(
                            self.track_plate_history[tid], key=lambda x: x[0]
                        )
                        best_plate = best_entry[1]
                        vehicle_obj = Vehicle(
                            bbox=[x1, y1, x2, y2, 0], id=tid, plate=best_plate
                        )
                        annotation_dict[tid] = {"vehicle": vehicle_obj}

                # Anota a imagem com as caixas dos veículos e com a detecção otimizada das placas (verde)
                annotated_frame = self.annotate_frame(
                    frame.copy(), annotation_dict, tracks
                )

                # Em debug, sobrepõe também todas as detecções de placas/OCR (amarelo)
                if debug:
                    annotated_frame = self._annotate_debug_plate_detections(
                        annotated_frame, frame_plate_inferences
                    )
                    debug_text = (
                        f"Veículo: {vehicle_detection_time:.1f}ms | "
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
                        cap.release()
                        if video_writer is not None:
                            video_writer.release()
                        cv2.destroyAllWindows()
                        pbar.close()
                        write_csv(final_frame_results, output_csv)
                        return final_frame_results

                if video_writer is not None:
                    video_writer.write(annotated_frame)

                # Verifica os tracks perdidos (não presentes no frame atual)
                lost_tracks = self.active_tracks - current_tracks
                for tid in lost_tracks:
                    if (
                        tid in self.track_plate_history
                        and self.track_plate_history[tid]
                    ):
                        best_entry = max(
                            self.track_plate_history[tid], key=lambda x: x[0]
                        )
                        best_plate = best_entry[1]
                        vehicle_bbox = best_entry[2]
                        vehicle_obj = Vehicle(
                            bbox=vehicle_bbox, id=tid, plate=best_plate
                        )
                        if frame_indices[i] not in final_frame_results:
                            final_frame_results[frame_indices[i]] = {}
                        final_frame_results[frame_indices[i]][tid] = {
                            "vehicle": vehicle_obj
                        }
                        del self.track_plate_history[tid]
                self.active_tracks = current_tracks

            frames_buffer = []
            frame_indices = []

        # Finaliza tracks ativos ao final do vídeo
        for tid in self.active_tracks:
            if tid in self.track_plate_history and self.track_plate_history[tid]:
                best_entry = max(self.track_plate_history[tid], key=lambda x: x[0])
                best_plate = best_entry[1]
                vehicle_bbox = best_entry[2]
                vehicle_obj = Vehicle(bbox=vehicle_bbox, id=tid, plate=best_plate)
                if frame_counter not in final_frame_results:
                    final_frame_results[frame_counter] = {}
                final_frame_results[frame_counter][tid] = {"vehicle": vehicle_obj}

        cap.release()
        if video_writer is not None:
            video_writer.release()
        pbar.close()
        if debug:
            cv2.destroyAllWindows()
        write_csv(final_frame_results, output_csv)
        return final_frame_results
