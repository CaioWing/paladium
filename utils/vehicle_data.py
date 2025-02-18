from dataclasses import dataclass
from typing import List, Optional
from utils.plate_data import LicensePlate


@dataclass
class Vehicle:
    "Estrutura para representar um veículo"
    bbox: List[float]  # [x1, y1, x2, y2, score]
    id: int
    label: str = "car"
    plate: Optional[LicensePlate] = None
    frame_nmr: int = 0  # Número do frame da detecção final
