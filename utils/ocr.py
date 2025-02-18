import string
import numpy as np
import cv2
from fast_plate_ocr import ONNXPlateRecognizer


class OptimizedOCR:
    def __init__(self):
        # Inicializa o modelo OCR
        self.model = ONNXPlateRecognizer("global-plates-mobile-vit-v2-model")
        self.dict_char_to_int = {
            "O": "0",
            "I": "1",
            "J": "3",
            "A": "4",
            "G": "6",
            "S": "5",
        }
        self.dict_int_to_char = {
            "0": "O",
            "1": "I",
            "3": "J",
            "4": "A",
            "6": "G",
            "5": "S",
        }

    def license_complies_format(self, text: str) -> bool:
        if len(text) != 7:
            return False
        return (
            (text[0] in string.ascii_uppercase or text[0] in self.dict_int_to_char)
            and (text[1] in string.ascii_uppercase or text[1] in self.dict_int_to_char)
            and (text[2] in "0123456789" or text[2] in self.dict_char_to_int)
            and (text[3] in "0123456789" or text[3] in self.dict_char_to_int)
            and (text[4] in string.ascii_uppercase or text[4] in self.dict_int_to_char)
            and (text[5] in string.ascii_uppercase or text[5] in self.dict_int_to_char)
            and (text[6] in string.ascii_uppercase or text[6] in self.dict_int_to_char)
        )

    def format_license(self, text: str) -> str:
        license_plate = ""
        mapping = {
            0: self.dict_int_to_char,
            1: self.dict_int_to_char,
            2: self.dict_char_to_int,
            3: self.dict_char_to_int,
            4: self.dict_int_to_char,
            5: self.dict_int_to_char,
            6: self.dict_int_to_char,
        }
        for i in range(7):
            if text[i] in mapping[i]:
                license_plate += mapping[i][text[i]]
            else:
                license_plate += text[i]
        return license_plate

    def read_license_plate(self, license_plate_crop):
        """
        Realiza OCR na região da placa.
        Agora, caso o crop seja um array NumPy (o formato esperado), passa-o diretamente para o modelo.
        Os prints de erro/inferência foram omitidos.
        """
        try:
            if license_plate_crop is None or license_plate_crop.size == 0:
                return "", 0.0

            result = self.model.run(license_plate_crop)
            if result and len(result) > 0:
                text = result[0]
                text = text.upper().replace(" ", "")
                if text != "":
                    return text, 1.0
        except Exception as e:
            print(e)
            pass

        return "", 0.0


_ocr_reader = None


def get_ocr_reader():
    global _ocr_reader
    if _ocr_reader is None:
        _ocr_reader = OptimizedOCR()
    return _ocr_reader


def read_license_plate(license_plate_crop):
    return get_ocr_reader().read_license_plate(license_plate_crop)
