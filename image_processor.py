# /paladium/image_processor.py

from ultralytics import YOLO
import cv2
import numpy as np
import torch
import yolov5
from utils.ocr import read_license_plate
from utils.preprocessing import preprocess_license_plate
from utils.plate_data import LicensePlate
from utils.vehicle_data import Vehicle


class ImageProcessor:
    def __init__(self):
        # Carrega o modelo principal de detecção de veículos (exportado para ONNX)
        self.vehicle_model = YOLO(
            "models/yolo11n.onnx", verbose=False
        )  # Modelo ONNX gerado via export()

        # Carrega o detector especializado para placas (YOLOv5)
        self.plate_model = yolov5.load("keremberke/yolov5n-license-plate")
        self.plate_model.conf = 0.4  # Limiar de confiança para NMS
        self.plate_model.iou = 0.45  # Limiar de IoU para NMS
        self.plate_model.agnostic = False
        self.plate_model.multi_label = False
        self.plate_model.max_det = 1000

        # Define o dispositivo para o modelo de placas
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.plate_model.to(self.device)

        # Mapeamento das classes de veículos (índices COCO)
        self.vehicle_classes = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

    def preprocess_image(self, image):
        """Converte a imagem BGR para RGB."""
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def detect_vehicles(self, frame):
        """
        Detecta veículos utilizando o modelo YOLO ONNX carregado.
        O modelo já realiza os pré-processamentos necessários e retorna resultados
        com caixas no formato [x1, y1, x2, y2, conf, cls].
        """
        # Executa a inferência
        results = self.vehicle_model(frame)
        vehicles = []
        if results and len(results) > 0:
            # Acessa as detecções no primeiro resultado
            boxes = results[
                0
            ].boxes.data  # Geralmente um tensor ou array com formato [N, 6]
            for det in boxes:
                # Converte a detecção para lista, se necessário
                det_list = det.tolist() if hasattr(det, "tolist") else list(det)
                x1, y1, x2, y2, conf, cls = det_list
                if int(cls) in self.vehicle_classes:
                    vehicles.append([x1, y1, x2, y2, conf])
        return vehicles

    def detect_license_plates(self, frame):
        """
        Detecta placas utilizando o detector especializado baseado em YOLOv5.
        """
        # Converte para RGB se necessário
        _ = self.preprocess_image(frame)
        with torch.no_grad():
            outputs = self.plate_model(frame, size=640)
        plates = []
        predictions = outputs.pred[0]
        boxes = predictions[:, :4]
        scores = predictions[:, 4].numpy()
        categories = predictions[:, 5]
        for score, _, box in zip(scores, categories, boxes):
            x1, y1, x2, y2 = box.cpu().numpy()
            plates.append([x1, y1, x2, y2, score, "plate"])
        return plates

    def annotate_image(self, frame, results):
        """
        Desenha retângulos e exibe o texto das placas na imagem.
        """
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

    def process_image(self, frame):
        """
        Processa uma imagem:
          1. Detecta veículos utilizando o modelo ONNX.
          2. Atribui um ID a cada veículo detectado.
          3. Detecta placas e associa a um veículo se a placa estiver contida na caixa do veículo.
          4. Retorna os resultados e a imagem anotada.
        """
        results = {}

        # Detecta veículos e cria objetos com ID
        vehicles = self.detect_vehicles(frame)
        vehicles_with_id = []
        for idx, vehicle in enumerate(vehicles):
            vehicles_with_id.append(vehicle + [idx])
        vehicle_objects = {
            vid: Vehicle(bbox=v[:5], id=vid) for v in vehicles_with_id for vid in [v[5]]
        }

        # Detecta placas e associa com o veículo cuja caixa as contém
        plates = self.detect_license_plates(frame)
        for plate in plates:
            p_x1, p_y1, p_x2, p_y2, p_score, _ = plate
            for vehicle in vehicles_with_id:
                x1, y1, x2, y2, score, vehicle_id = vehicle
                if vehicle_objects[vehicle_id].plate is not None:
                    continue
                if p_x1 >= x1 and p_y1 >= y1 and p_x2 <= x2 and p_y2 <= y2:
                    license_plate_crop = frame[
                        int(p_y1) : int(p_y2), int(p_x1) : int(p_x2)
                    ]
                    processed_plate = preprocess_license_plate(license_plate_crop)
                    text, text_score = read_license_plate(processed_plate)
                    if text:
                        lp = LicensePlate(
                            bbox=[p_x1, p_y1, p_x2, p_y2],
                            text=text,
                            detection_confidence=p_score,
                            ocr_confidence=text_score,
                        )
                        vehicle_objects[vehicle_id].plate = lp
                        break

        for vid, vehicle_obj in vehicle_objects.items():
            if vehicle_obj.plate is not None:
                results[vid] = {"vehicle": vehicle_obj}

        annotated_image = self.annotate_image(frame.copy(), results)
        return results, annotated_image
