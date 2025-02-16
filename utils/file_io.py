import csv
from utils.plate_data import LicensePlate
from utils.vehicle_data import Vehicle


def write_csv(results, output_path):
    """
    Escreve os resultados em um arquivo CSV com o seguinte cabeçalho:
      frame_nmr, car_id, car_bbox, license_plate_bbox, license_plate_bbox_score, license_number, license_number_score
    """
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "frame_nmr",
                "car_id",
                "car_bbox",
                "license_plate_bbox",
                "license_plate_bbox_score",
                "license_number",
                "license_number_score",
            ]
        )
        for frame_nmr, data in results.items():
            for car_id, info in data.items():
                if (
                    "vehicle" in info
                    and isinstance(info["vehicle"], Vehicle)
                    and info["vehicle"].plate is not None
                ):
                    vehicle = info["vehicle"]
                    plate = vehicle.plate
                    # Usa as 4 primeiras coordenadas do bbox do veículo
                    car_bbox = vehicle.bbox[:4]
                    license_plate_bbox = plate.bbox
                    writer.writerow(
                        [
                            frame_nmr,
                            car_id,
                            f"[{' '.join(str(x) for x in car_bbox)}]",
                            f"[{' '.join(str(x) for x in license_plate_bbox)}]",
                            plate.detection_confidence,
                            plate.text,
                            plate.ocr_confidence,
                        ]
                    )
