from dataclasses import dataclass
from typing import List


@dataclass
class LicensePlate:
    bbox: List[float]  # [x1, y1, x2, y2]
    text: str
    detection_confidence: float = 0.0  # Score da detecção da placa (SAHI)
    ocr_confidence: float = 0.0  # Score do OCR
