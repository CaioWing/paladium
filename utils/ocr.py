# utils/ocr.py
import string
import numpy as np
import cv2
from fast_plate_ocr import ONNXPlateRecognizer


class OptimizedOCR:
    def __init__(self):
        # Initialize the ONNX-based plate recognizer
        self.model = ONNXPlateRecognizer("global-plates-mobile-vit-v2-model")

        # Dictionaries for character conversion (Mercosul format)
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
        """
        Verifies if the license plate text follows Mercosul format:
        - 7 characters: positions 0-1 letters, 2-3 digits, 4-6 letters
        Allows conversions using dictionaries.
        """
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
        """Formats the license plate text applying defined conversions."""
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
        Performs OCR on the preprocessed license plate region using fast-plate-ocr.
        Returns formatted text and confidence score.
        """
        try:
            # Check if the crop is valid (non-empty)
            if license_plate_crop is None or license_plate_crop.size == 0:
                print("Empty license plate crop received, skipping OCR.")
                return "", 0.0

            # Ensure the image is in the correct format
            if isinstance(license_plate_crop, np.ndarray):
                temp_path = "temp_plate.jpg"
                success = cv2.imwrite(temp_path, license_plate_crop)
                if not success:
                    print("Failed to write temporary image for OCR.")
                    return "", 0.0
                result = self.model.run(temp_path)
            else:
                result = self.model.run(license_plate_crop)

            # Extract text and confidence from result
            if result and len(result) > 0:
                text = result[0]  # Get the text from the first result
                # confidence = result[0].confidence  # Get the confidence score

                # Clean and format text
                text = text.upper().replace(" ", "")

                # For mercosul vehicles
                # if self.license_complies_format(text):
                #     formatted_text = self.format_license(text)
                #     return formatted_text, 1.0

                if text != "":
                    return text, 1.0

        except Exception as e:
            print(f"Error in OCR processing: {e}")

        return "", 0.0


# Global instance
ocr_reader = OptimizedOCR()


# Function to maintain compatibility with existing code
def read_license_plate(license_plate_crop):
    return ocr_reader.read_license_plate(license_plate_crop)
