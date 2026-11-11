"""
Dynamic Risk Engine & Simple Explainable Risk Level Formulator for SafetyVision AI.

Risk Classification Logic (College Capstone Demonstration Standard):
--------------------------------------------------------------------
- LOW: All required PPE present (Hardhat ✓, Mask ✓, Safety Vest ✓). Fully compliant.
- MEDIUM: 1 PPE item missing (e.g. Missing Hardhat OR Missing Vest OR Missing Mask).
- HIGH: 2 PPE items missing OR 1 PPE item missing near machinery / vehicle hazard.
- CRITICAL: Multiple PPE violations combined with active machinery or vehicle proximity hazard.

Compliance Formula:
------------------
Each item = 33.33%
Hardhat (33.33%) + Mask (33.33%) + Safety Vest (33.34%) = 100.0%
"""

def get_risk_level_from_score(score):
    if score >= 81:
        return 'CRITICAL'
    elif score >= 61:
        return 'HIGH'
    elif score >= 31:
        return 'MEDIUM'
    else:
        return 'LOW'

class DynamicRiskEngine:
    def __init__(self):
        pass

    def evaluate_person_risk(self, person, zone_multiplier=1.0):
        """
        Calculates simple, transparent, explainable risk score and status for a TrackedPerson.
        """
        missing_count = 0
        missing_items = []

        if person.ppe_status.get('Hardhat', '').upper() == 'MISSING':
            missing_count += 1
            missing_items.append('Hardhat')
        if person.ppe_status.get('Mask', '').upper() == 'MISSING':
            missing_count += 1
            missing_items.append('Mask')
        if person.ppe_status.get('Safety Vest', '').upper() == 'MISSING':
            missing_count += 1
            missing_items.append('Safety Vest')

        near_hazard = getattr(person, 'near_machinery', False) or getattr(person, 'near_vehicle', False)
        is_restricted = 'Restricted Zone Breach' in person.active_violations

        # Simple, deterministic evaluation
        if missing_count == 0 and not is_restricted:
            risk_level = 'LOW'
            status = 'COMPLIANT'
            risk_score = 0.0
            risk_reason = 'Fully Compliant - All Mandated PPE Present'
        elif missing_count == 1:
            status = 'PPE VIOLATION'
            if near_hazard or is_restricted:
                risk_level = 'HIGH'
                risk_score = 70.0
                risk_reason = f"Missing {missing_items[0]} near active equipment/hazard"
            else:
                risk_level = 'MEDIUM'
                risk_score = 45.0
                risk_reason = f"Missing {missing_items[0]}"
        elif missing_count == 2:
            if near_hazard or is_restricted:
                risk_level = 'CRITICAL'
                status = 'CRITICAL PPE VIOLATION'
                risk_score = 90.0
                risk_reason = f"Multiple PPE missing ({', '.join(missing_items)}) near hazard"
            else:
                risk_level = 'HIGH'
                status = 'PPE VIOLATION'
                risk_score = 75.0
                risk_reason = f"Multiple PPE missing: {', '.join(missing_items)}"
        else: # missing_count >= 3 or severe breach
            risk_level = 'CRITICAL'
            status = 'CRITICAL PPE VIOLATION'
            risk_score = 95.0 if not near_hazard else 100.0
            risk_reason = f"Critical non-compliance: {', '.join(missing_items) if missing_items else 'Unprotected worker'}"

        person.risk_score = round(risk_score, 1)
        person.risk_level = risk_level
        person.status = status
        person.risk_reason = risk_reason

        return person.risk_score, person.risk_level

    @staticmethod
    def calculate_site_safety_score(avg_risk, compliance_rate, critical_count, total_incidents):
        """
        Overall Site Safety Score (0-100). Transparent and deterministic.
        """
        score = 100.0 - (0.4 * avg_risk + 0.3 * (100.0 - compliance_rate) + 0.3 * min(100.0, critical_count * 15.0))
        return round(min(100.0, max(0.0, score)), 1)
