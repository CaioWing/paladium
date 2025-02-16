def get_car(license_plate, track_ids):
    """Associa uma placa a um veículo (adaptado para imagem/vídeo)."""
    x1, y1, x2, y2 = license_plate[:4]
    for track_id in track_ids:
        xcar1, ycar1, xcar2, ycar2, _ = track_id
        if x1 > xcar1 and y1 > ycar1 and x2 < xcar2 and y2 < ycar2:
            return track_id
    return None
