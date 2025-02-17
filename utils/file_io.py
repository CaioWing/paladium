import csv
from utils.plate_data import LicensePlate
from utils.vehicle_data import Vehicle


def write_csv(results, output_path):
    """
    Escreve os resultados finais em um arquivo CSV com o cabeçalho:
      frame_nmr, car_id, car_bbox, license_plate_bbox, license_plate_bbox_score, license_number, license_number_score
    Cada linha representa um veículo único.
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
        for track_id, data in results.items():
            if (
                "vehicle" in data
                and isinstance(data["vehicle"], Vehicle)
                and data["vehicle"].plate is not None
            ):
                vehicle = data["vehicle"]
                plate = vehicle.plate
                # Usa as 4 primeiras coordenadas do bbox do veículo
                car_bbox = vehicle.bbox[:4]
                license_plate_bbox = plate.bbox
                writer.writerow(
                    [
                        vehicle.frame_nmr,
                        track_id,
                        f"[{' '.join(str(x) for x in car_bbox)}]",
                        f"[{' '.join(str(x) for x in license_plate_bbox)}]",
                        plate.detection_confidence,
                        plate.text,
                        plate.ocr_confidence,
                    ]
                )
