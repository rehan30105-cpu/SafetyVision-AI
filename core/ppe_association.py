"""
SAFETYVISION AI - Person-PPE Association Engine
Performs anatomically guided spatial association, class-specific confidence validation,
conflict resolution, and 3-state temporal aggregation ('PRESENT', 'MISSING', 'UNCERTAIN')
between raw YOLO detections and tracked individuals.
"""

def point_in_box(px, py, box):
    return box[0] <= px <= box[2] and box[1] <= py <= box[3]

def get_intersection(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    if xB <= xA or yB <= yA:
        return 0.0
    return (xB - xA) * (yB - yA)

def box_area(box):
    return max(0.0, (box[2] - box[0])) * max(0.0, (box[3] - box[1]))

class PPEAssociationEngine:
    def __init__(self, hardhat_weight=33.33, mask_weight=33.33, vest_weight=33.34, **kwargs):
        self.categories = ['Hardhat', 'Mask', 'Safety Vest']
        self.weights = {
            'Hardhat': hardhat_weight,
            'Mask': mask_weight,
            'Safety Vest': vest_weight
        }

        # Class-specific confidence thresholds to prevent false positives (e.g. cloth as hardhat)
        self.class_thresholds = {
            'Hardhat': 0.55,       # Stricter: ensures real hardhat, filters soft cloth/caps
            'NO-Hardhat': 0.40,
            'Mask': 0.42,
            'NO-Mask': 0.40,
            'Safety Vest': 0.45,
            'NO-Safety Vest': 0.40
        }

    def associate(self, tracked_persons, ppe_detections):
        """
        Associates PPE detections with tracked persons using anatomical priors,
        confidence validation, and conflict resolution.
        """
        if not tracked_persons:
            return tracked_persons

        # Reset per-person candidate detections for this instant frame
        for person in tracked_persons:
            person.associated_ppe_items = []

        # Anatomical vertical bounds (relative to person box height H)
        # Person box: y1 (top) to y2 (bottom)
        anatomical_rules = {
            'Hardhat': {
                'classes': ['Hardhat', 'NO-Hardhat'],
                'y_range': (-0.20, 0.30), # Head region (strictly top 30% of body)
                'x_margin': 0.15
            },
            'Mask': {
                'classes': ['Mask', 'NO-Mask'],
                'y_range': (0.05, 0.45),  # Face/neck region
                'x_margin': 0.15
            },
            'Safety Vest': {
                'classes': ['Safety Vest', 'NO-Safety Vest'],
                'y_range': (0.18, 0.85),  # Torso/chest region
                'x_margin': 0.15
            }
        }

        # Step 1: For each PPE detection, validate threshold and match to the correct person
        for det in ppe_detections:
            cname = det.get('class_name', '')
            iconf = det.get('confidence', 0.0)
            ibox = det.get('bbox', [0, 0, 0, 0])
            iarea = box_area(ibox)
            if iarea <= 0:
                continue

            # Class-specific threshold filtering
            min_thresh = self.class_thresholds.get(cname, 0.40)
            if iconf < min_thresh:
                continue

            icx = (ibox[0] + ibox[2]) / 2.0
            icy = (ibox[1] + ibox[3]) / 2.0

            # Determine gear category
            gear_cat = None
            for cat, rule in anatomical_rules.items():
                if cname in rule['classes']:
                    gear_cat = cat
                    break

            if not gear_cat:
                continue

            rule = anatomical_rules[gear_cat]
            y_min_ratio, y_max_ratio = rule['y_range']
            x_margin = rule['x_margin']

            best_person = None
            best_score = -1.0

            for person in tracked_persons:
                pbox = person.bbox
                pw = max(1.0, pbox[2] - pbox[0])
                ph = max(1.0, pbox[3] - pbox[1])

                reg_top = pbox[1] + y_min_ratio * ph
                reg_bottom = pbox[1] + y_max_ratio * ph
                reg_left = pbox[0] - x_margin * pw
                reg_right = pbox[2] + x_margin * pw
                reg_box = [reg_left, reg_top, reg_right, reg_bottom]

                # Center must fall within anatomical sub-region
                in_region = point_in_box(icx, icy, reg_box)

                # Overlap with person box
                inter = get_intersection(ibox, pbox)
                containment = inter / iarea if iarea > 0 else 0.0

                if in_region and containment >= 0.20:
                    horiz_dist = abs(icx - (pbox[0] + pbox[2]) / 2.0) / pw
                    alignment_score = max(0.0, 1.0 - horiz_dist)
                    score = (containment * 2.0) + alignment_score + 2.0

                    if score > best_score:
                        best_score = score
                        best_person = person

            # Associate item with best matching person
            if best_person and best_score > 1.2:
                item_copy = dict(det)
                item_copy['gear_category'] = gear_cat
                best_person.associated_ppe_items.append(item_copy)

        # Step 2: For each person, evaluate gear status across the 3 categories
        for person in tracked_persons:
            person.active_violations = []

            # Group items by category
            items_by_cat = {'Hardhat': [], 'Mask': [], 'Safety Vest': []}
            for it in person.associated_ppe_items:
                cat = it.get('gear_category')
                if cat in items_by_cat:
                    items_by_cat[cat].append(it)

            for cat in self.categories:
                candidates = items_by_cat[cat]

                if not candidates:
                    # Neither positive nor negative detected in this frame.
                    # Record observation: if person previously confirmed present, hold grace
                    prev_status = person.ppe_status.get(cat, 'UNCERTAIN')
                    if prev_status == 'PRESENT' and person.ppe_grace.get(cat, 0) < 3:
                        obs_state = 'PRESENT'
                        details = person.ppe_details.get(cat)
                    else:
                        obs_state = 'MISSING'
                        details = {
                            'status': 'MISSING',
                            'label': '✗ MISSING',
                            'detected_class': f"NO-{cat}",
                            'confidence': 0.0
                        }
                    person.record_ppe_observation(cat, obs_state, details)
                    continue

                pos_items = [it for it in candidates if not it['class_name'].startswith('NO-')]
                neg_items = [it for it in candidates if it['class_name'].startswith('NO-')]

                best_pos = max(pos_items, key=lambda x: x['confidence']) if pos_items else None
                best_neg = max(neg_items, key=lambda x: x['confidence']) if neg_items else None

                # Conflict resolution: compare confidences
                if best_pos and best_neg:
                    # Positive must convincingly exceed negative
                    if best_pos['confidence'] > (best_neg['confidence'] + 0.10):
                        chosen = best_pos
                        state = 'PRESENT'
                    elif best_neg['confidence'] >= best_pos['confidence']:
                        chosen = best_neg
                        state = 'MISSING'
                    else:
                        chosen = best_neg
                        state = 'UNCERTAIN'
                elif best_pos:
                    chosen = best_pos
                    state = 'PRESENT'
                else:
                    chosen = best_neg
                    state = 'MISSING'

                chosen_conf = round(chosen['confidence'] * 100.0, 1) if chosen['confidence'] <= 1.0 else round(chosen['confidence'], 1)
                details = {
                    'status': state,
                    'label': '✓ DETECTED' if state == 'PRESENT' else ('✗ MISSING' if state == 'MISSING' else '? UNCERTAIN'),
                    'detected_class': chosen['class_name'],
                    'confidence': chosen_conf
                }

                person.record_ppe_observation(cat, state, details)

            # Compile instant compliance & active violations
            present_count = sum(1 for cat in self.categories if person.ppe_status[cat] == 'PRESENT')
            person.compliance_score = round((present_count / 3.0) * 100.0, 1)

            person.active_violations = [f"Missing {cat}" for cat in self.categories if person.ppe_status[cat] == 'MISSING']

            if present_count == 3:
                person.status = 'COMPLIANT'
            else:
                person.status = 'PPE VIOLATION'

        return tracked_persons
