from dataclasses import dataclass
from typing import List, Optional
from utils.plate_data import LicensePlate


@dataclass
class Vehicle:
    bbox: List[float]  # [x1, y1, x2, y2, score]
    id: int
    label: str = "car"
    plate: Optional[LicensePlate] = None
