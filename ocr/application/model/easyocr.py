import easyocr
from application.model.modelbase import ModelBase

class EasyOCR(ModelBase):
    def __init__(self):
        super().__init__("EasyOCR")
        self.reader = easyocr.Reader(['en'])

    def perform_ocr(self, input_path, output_path):
        try:
            result = self.reader.readtext(input_path)
            chars_in_current_line = 0
            max_line_width = 80
            with open(output_path, "w") as f:
                for entry in result:
                    seq = entry[1]
                    chars_in_current_line += len(seq)
                    if chars_in_current_line > max_line_width:
                        f.writelines(f"{seq}\n")
                        chars_in_current_line = 0
                    else:
                        f.writelines(f"{seq} ")
            return True
        except Exception:
            return False