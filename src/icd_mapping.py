# TODO

class ICDMapEncoder:
    def __init__(self, icd_map):
        self.icd_map = icd_map

    def encode(self, icd_code):
        return self.icd_map.get(icd_code, 0)  # Default to 0 if code not found