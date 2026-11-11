"""SAFETYVISION AI — Canonical PPE violation counting.
Single source of truth:
- ppe_violations   = individually missing required PPE items (Hardhat/Mask/Safety Vest)
- non_compliant    = people missing >= 1 required item
- incidents        = confirmed episodes (existing incident rules, untouched)
Only 'MISSING' counts; 'PRESENT'/'UNCERTAIN'/unknown never count.
"""

REQUIRED_PPE_ITEMS = ('Hardhat', 'Mask', 'Safety Vest')

_KEY_ALIASES = {
    'hardhat': 'Hardhat', 'hh': 'Hardhat',
    'mask': 'Mask', 'mk': 'Mask',
    'vest': 'Safety Vest', 'vt': 'Safety Vest',
    'safety vest': 'Safety Vest', 'safety_vest': 'Safety Vest',
}


def _norm_state(value):
    return str(value or '').strip().upper()


def _person_state(person, item):
    if not isinstance(person, dict):
        return ''
    if item in person:
        return _norm_state(person.get(item))
    low = item.lower()
    if low in person:
        return _norm_state(person.get(low))
    for alias, canonical in _KEY_ALIASES.items():
        if canonical == item and alias in person:
            return _norm_state(person.get(alias))
    return ''


def count_person_ppe_violations(person):
    """Missing required PPE items for one person (0-3). Only 'MISSING' counts."""
    count = 0
    for item in REQUIRED_PPE_ITEMS:
        if _person_state(person, item) == 'MISSING':
            count += 1
    return count


def is_person_fully_equipped(person):
    """True only when Hardhat, Mask, and Safety Vest are all confirmed PRESENT."""
    return all(_person_state(person, item) == 'PRESENT' for item in REQUIRED_PPE_ITEMS)


def summarize_ppe(person_results):
    """Aggregate using the canonical rule.

    - ppe_violations: individually missing required items (MISSING only;
      PRESENT/UNCERTAIN/unknown never count).
    - compliant: people with all three items confirmed PRESENT.
    - non_compliant_people: everyone else (missing >= 1 item, or
      unknown/unconfirmed state preserved without inflating violations).
    """
    persons = person_results or []
    people = len(persons)
    item_violations = sum(count_person_ppe_violations(p) for p in persons)
    compliant = sum(1 for p in persons if is_person_fully_equipped(p))
    return {
        'people': people,
        'compliant': compliant,
        'non_compliant_people': people - compliant,
        'ppe_violations': item_violations,
    }
