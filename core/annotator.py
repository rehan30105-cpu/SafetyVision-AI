"""
SAFETYVISION AI - Visual Frame & Image Annotation Engine
Draws high-contrast, evaluator-ready annotations directly onto images and video frames.
Displays person outer boxes with status & compliance, plus inner PPE detection boxes
with explicit class names and confidence percentages (e.g., Hardhat 91%, NO-Hardhat 86%).
"""

import cv2
import numpy as np

def draw_person_annotations(frame, person_results, raw_detections=None):
    """
    frame: numpy BGR image array
    person_results: list of person result dicts (from TrackedPerson.to_person_result)
    raw_detections: list of all raw YOLO detections
    Returns: annotated numpy BGR image array
    """
    annotated = frame.copy()
    h, w = annotated.shape[:2]

    # Color palette (BGR for OpenCV)
    COLOR_SAFE = (129, 185, 16)      # #10B981 Emerald Green
    COLOR_VIOLATION = (68, 68, 239)   # #EF4444 Vivid Red
    COLOR_AMBER = (11, 158, 245)      # #F59E0B Amber
    COLOR_ORANGE = (22, 115, 249)     # #F97316 Orange
    COLOR_CYAN = (212, 182, 6)        # #06B6D4 Cyan
    COLOR_BLUE = (246, 130, 59)       # #3B82F6 Blue
    COLOR_HAZARD = (153, 72, 236)     # #EC4899 Magenta
    COLOR_VEHICLE = (241, 102, 99)    # #6366F1 Indigo

    # 1. Draw Non-Person Environmental Hazards (Machinery, Vehicles, Safety Cones)
    if raw_detections:
        for det in raw_detections:
            cname = det.get('class_name', '')
            if cname in ['Person', 'Hardhat', 'Mask', 'Safety Vest', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest']:
                continue

            box = [int(v) for v in det.get('bbox', [0, 0, 0, 0])]
            x1, y1, x2, y2 = max(0, box[0]), max(0, box[1]), min(w, box[2]), min(h, box[3])
            conf_pct = int(round(det.get('confidence', 0.8) * 100))

            if cname == 'Safety Cone':
                color = COLOR_AMBER
                label_text = f"Safety Cone {conf_pct}%"
            elif cname == 'machinery':
                color = COLOR_HAZARD
                label_text = f"Machinery Hazard {conf_pct}%"
            elif cname == 'vehicle':
                color = COLOR_VEHICLE
                label_text = f"Industrial Vehicle {conf_pct}%"
            else:
                color = (200, 200, 200)
                label_text = f"{cname} {conf_pct}%"

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            _draw_tag(annotated, label_text, x1, max(18, y1 - 4), color, text_color=(255, 255, 255), font_scale=0.45)

    # 2. Draw PPE Bounding Boxes (Both Associated & Unassociated)
    # Collect all drawn PPE bounding boxes to avoid duplicate drawing
    drawn_ppe_boxes = set()

    # Draw PPE associated with each person
    for p in person_results:
        associated_items = p.get('associated_ppe', [])
        for item in associated_items:
            ibox = [int(v) for v in item.get('bbox', [0, 0, 0, 0])]
            box_key = (ibox[0], ibox[1], ibox[2], ibox[3])
            drawn_ppe_boxes.add(box_key)

            cname = item.get('class_name', '')
            conf_pct = int(round(item.get('confidence', 0.0) * 100))
            is_neg = cname.startswith('NO-')

            # Box color
            box_color = COLOR_VIOLATION if is_neg else (COLOR_CYAN if 'Mask' in cname else (COLOR_SAFE if 'Hardhat' in cname else COLOR_BLUE))
            label_text = f"{cname} {conf_pct}%"

            ix1, iy1, ix2, iy2 = max(0, ibox[0]), max(0, ibox[1]), min(w, ibox[2]), min(h, ibox[3])
            cv2.rectangle(annotated, (ix1, iy1), (ix2, iy2), box_color, 2)
            _draw_tag(annotated, label_text, ix1, min(h - 4, iy2 + 14), box_color, text_color=(255, 255, 255), font_scale=0.40)

    # If raw_detections has PPE not associated to any person (e.g. unassigned gear), draw it too
    if raw_detections:
        for det in raw_detections:
            cname = det.get('class_name', '')
            if cname in ['Hardhat', 'NO-Hardhat', 'Mask', 'NO-Mask', 'Safety Vest', 'NO-Safety Vest']:
                ibox = [int(v) for v in det.get('bbox', [0, 0, 0, 0])]
                box_key = (ibox[0], ibox[1], ibox[2], ibox[3])
                if box_key not in drawn_ppe_boxes:
                    conf_pct = int(round(det.get('confidence', 0.0) * 100))
                    is_neg = cname.startswith('NO-')
                    box_color = COLOR_VIOLATION if is_neg else COLOR_SAFE
                    label_text = f"{cname} {conf_pct}%"
                    ix1, iy1, ix2, iy2 = max(0, ibox[0]), max(0, ibox[1]), min(w, ibox[2]), min(h, ibox[3])
                    cv2.rectangle(annotated, (ix1, iy1), (ix2, iy2), box_color, 2)
                    _draw_tag(annotated, label_text, ix1, min(h - 4, iy2 + 14), box_color, text_color=(255, 255, 255), font_scale=0.40)

    # 3. Draw Person Outer Bounding Boxes with Visual Hierarchy
    for p in person_results:
        box = [int(v) for v in p.get('bbox', [0, 0, 0, 0])]
        x1, y1, x2, y2 = max(0, box[0]), max(0, box[1]), min(w, box[2]), min(h, box[3])

        compliance = p.get('compliance', 0.0)
        is_compliant = (compliance >= 99.0)
        risk = p.get('risk', 'LOW')

        # Person box border color based on compliance & risk
        if is_compliant:
            border_color = COLOR_SAFE
        elif risk == 'CRITICAL':
            border_color = COLOR_VIOLATION
        elif risk == 'HIGH':
            border_color = COLOR_ORANGE
        else:
            border_color = COLOR_AMBER

        # Draw Person Outer Box (3px thick)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), border_color, 3)

        # Header Label: Person #1 | PPE: 66.7% | VIOLATION
        label = p.get('label', f"Person #{p.get('person_id', 1)}")
        status_text = 'COMPLIANT' if is_compliant else 'VIOLATION'
        header_text = f"{label} | PPE: {compliance}% | {status_text}"

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.52
        thickness = 2
        (tw, th), _ = cv2.getTextSize(header_text, font, font_scale, thickness)

        bg_y1 = max(0, y1 - th - 10)
        bg_y2 = y1
        bg_x2 = min(w, x1 + tw + 14)
        cv2.rectangle(annotated, (x1, bg_y1), (bg_x2, bg_y2), border_color, -1)
        cv2.putText(annotated, header_text, (x1 + 6, bg_y2 - 5),
                    font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

        # Sub-Tag if violation: Missing Gear Summary
        missing = p.get('missing_items', [])
        if not is_compliant and missing:
            missing_text = f"Missing: {', '.join(missing)}"
            (mw, mh), _ = cv2.getTextSize(missing_text, font, 0.42, 1)
            sub_y1 = max(0, bg_y1 - mh - 6)
            sub_y2 = bg_y1
            sub_x2 = min(w, x1 + mw + 12)
            cv2.rectangle(annotated, (x1, sub_y1), (sub_x2, sub_y2), (20, 24, 38), -1)
            cv2.putText(annotated, missing_text, (x1 + 6, sub_y2 - 3),
                        font, 0.42, (180, 180, 255), 1, cv2.LINE_AA)

    return annotated

def _draw_tag(img, text, x, y, bg_color, text_color=(255, 255, 255), font_scale=0.45):
    """Draws a clean, readable text tag with solid background box."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, 1)
    h, w = img.shape[:2]
    x1 = max(0, min(w - tw - 8, x))
    y1 = max(0, y - th - 4)
    x2 = min(w, x1 + tw + 8)
    y2 = min(h, y1 + th + 6)
    cv2.rectangle(img, (x1, y1), (x2, y2), bg_color, -1)
    cv2.putText(img, text, (x1 + 4, y2 - 3), font, font_scale, text_color, 1, cv2.LINE_AA)
