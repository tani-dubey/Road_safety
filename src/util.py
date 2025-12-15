import string
import easyocr
import csv
import os

# Initialize the OCR reader
reader = easyocr.Reader(['en'], gpu=False)

# Mapping dictionaries for character conversion
dict_char_to_int = {'O': '0',
                    'I': '1',
                    'J': '3',
                    'A': '4',
                    'G': '6',
                    'S': '5'}

dict_int_to_char = {'0': 'O',
                    '1': 'I',
                    '3': 'J',
                    '4': 'A',
                    '6': 'G',
                    '5': 'S'}


def write_csv(results, output_path):
    """
    Write results (with timestamps) to CSV.
    Appends new detections and creates the file if it doesn't exist.
    """
    rows = []
    for frame_nmr, cars in results.items():
        for car_id, data in cars.items():
            lp = data.get('license_plate', {})
            car = data.get('car', {})
            det_time = data.get('detection_time', {})

            # Only write valid detections with recognized text
            if 'text' in lp:
                rows.append({
                    'frame_nmr': frame_nmr,
                    'car_id': car_id,
                    'car_bbox': f"[{car['bbox'][0]} {car['bbox'][1]} {car['bbox'][2]} {car['bbox'][3]}]" if 'bbox' in car else '',
                    'license_plate_bbox': f"[{lp['bbox'][0]} {lp['bbox'][1]} {lp['bbox'][2]} {lp['bbox'][3]}]" if 'bbox' in lp else '',
                    'license_plate_bbox_score': lp.get('bbox_score', ''),
                    'license_number': lp.get('text', ''),
                    'license_number_score': lp.get('text_score', ''),
                    'timestamp_ms': det_time.get('time_ms', ''),
                    'timestamp_utc': det_time.get('time_utc', '')
                })

    fieldnames = [
        'frame_nmr',
        'car_id',
        'car_bbox',
        'license_plate_bbox',
        'license_plate_bbox_score',
        'license_number',
        'license_number_score',
        'timestamp_ms',
        'timestamp_utc'
    ]

    file_exists = os.path.exists(output_path)
    with open(output_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)


def license_complies_format(text):
    # if len(text) != 7:
    #     return False

    # if (text[0] in string.ascii_uppercase or text[0] in dict_int_to_char.keys()) and \
    #    (text[1] in string.ascii_uppercase or text[1] in dict_int_to_char.keys()) and \
    #    (text[2] in ['0','1','2','3','4','5','6','7','8','9'] or text[2] in dict_char_to_int.keys()) and \
    #    (text[3] in ['0','1','2','3','4','5','6','7','8','9'] or text[3] in dict_char_to_int.keys()) and \
    #    (text[4] in string.ascii_uppercase or text[4] in dict_int_to_char.keys()) and \
    #    (text[5] in string.ascii_uppercase or text[5] in dict_int_to_char.keys()) and \
    #    (text[6] in string.ascii_uppercase or text[6] in dict_int_to_char.keys()):
    #     return True
    # else:
    #     return False
    return len(text) >= 4


# def format_license(text):
#     license_plate_ = ''
#     mapping = {0: dict_int_to_char, 1: dict_int_to_char, 4: dict_int_to_char, 5: dict_int_to_char, 6: dict_int_to_char,
#                2: dict_char_to_int, 3: dict_char_to_int}
#     for j in [0, 1, 2, 3, 4, 5, 6]:
#         if text[j] in mapping[j].keys():
#             license_plate_ += mapping[j][text[j]]
#         else:
#             license_plate_ += text[j]
#     return license_plate_
def format_license(text):
    """
    Safely format license plate text.
    Works even if OCR returns short strings.
    """
    license_plate_ = ''
    mapping = {
        0: dict_int_to_char,
        1: dict_int_to_char,
        2: dict_char_to_int,
        3: dict_char_to_int,
        4: dict_int_to_char,
        5: dict_int_to_char,
        6: dict_int_to_char
    }

    max_len = min(len(text), 7)

    for j in range(max_len):
        char = text[j]
        if j in mapping and char in mapping[j]:
            license_plate_ += mapping[j][char]
        else:
            license_plate_ += char

    return license_plate_



def read_license_plate(license_plate_crop):
    # detections = reader.readtext(license_plate_crop)
    # for detection in detections:
    #     bbox, text, score = detection
    #     text = text.upper().replace(' ', '')
    #     if license_complies_format(text):
    #         return format_license(text), score
    # return None, None
    h, w = license_plate_crop.shape[:2]
    if h == 0 or w == 0:
        return None, None

    detections = reader.readtext(license_plate_crop)
    if not detections:
        return None, None

    for detection in detections:
        bbox, text, score = detection
        text = text.upper().replace(' ', '')
        print(f"🔹 OCR raw text: '{text}' with score {score}")  # log every OCR attempt
        if license_complies_format(text):
            return format_license(text), score

    return None, None


def get_car(license_plate, vehicle_track_ids):
    x1, y1, x2, y2, score, class_id = license_plate
    for j in range(len(vehicle_track_ids)):
        xcar1, ycar1, xcar2, ycar2, car_id = vehicle_track_ids[j]
        if x1 > xcar1 and y1 > ycar1 and x2 < xcar2 and y2 < ycar2:
            return xcar1, ycar1, xcar2, ycar2, car_id
    return -1, -1, -1, -1, -1
