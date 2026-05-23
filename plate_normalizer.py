import re

# Indian state codes for validation
INDIA_STATE_CODES = {
    "AN", "AP", "AR", "AS", "BR", "CG", "CH", "DD", "DL", "DN",
    "GA", "GJ", "HR", "HP", "JH", "JK", "KA", "KL", "LA", "LD",
    "MH", "ML", "MN", "MP", "MZ", "NL", "OD", "PB", "PY", "RJ",
    "SK", "TN", "TR", "TS", "UK", "UP", "WB", "BH"
}

# Safe character corrections (only visually identical)
OCR_CHAR_FIX = {
    "O": "0", "Q": "0", "I": "1", "Z": "2",
    "S": "5", "G": "6", "B": "8", "D": "0"
}


def validate_state_code(text: str) -> bool:
    """Return True if the first two characters are a valid Indian state code."""
    if not text or len(text) < 2:
        return False
    t = re.sub(r'[^A-Za-z0-9]', '', text).upper()
    if len(t) < 2:
        return False
    return t[:2] in INDIA_STATE_CODES


def is_valid_indian_plate(text: str) -> tuple[bool, str]:
    """Validate Indian plate format and return (is_valid, status).
    Status is one of: 'FULL', 'PARTIAL', 'INVALID'.
    """
    if not text:
        return False, "INVALID"

    t = re.sub(r'[^A-Za-z0-9]', '', text).upper()

    if len(t) < 2:
        return False, "INVALID"

    has_letters = any(c.isalpha() for c in t)
    has_numbers = any(c.isdigit() for c in t)
    if not (has_letters and has_numbers):
        return False, "INVALID"

    full_patterns = [
        r'^[A-Z]{2}[0-9]{2}[A-Z]{1,3}[0-9]{4}$',
        r'^[A-Z]{2}[0-9]{2}[A-Z]{1,2}[0-9]{3,4}$',
        r'^[A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{4}$',
        r'^BH[0-9]{2}[A-Z]{2}[0-9]{4}$',
    ]

    for pattern in full_patterns:
        if re.match(pattern, t):
            if not t.startswith('BH') and (len(t) < 2 or t[:2] not in INDIA_STATE_CODES):
                continue
            return True, "FULL"

    partial_patterns = [
        r'^[A-Z]{2}[0-9]{2}[A-Z]{1,3}$',
        r'^[A-Z]{2}[0-9]{2}[A-Z]{1,2}$',
        r'^[A-Z]{2}[0-9]{2}$',
        r'^[A-Z]{2}[0-9]{1}[A-Z]{1,2}$',
        r'^[A-Z]{2}[0-9]{1}[A-Z]{1,3}$',
        r'^[A-Z]{2}[0-9]{4,5}$',
        r'^[A-Z]{2}[0-9]{3,4}$',
        r'^[A-Z]{1,2}[0-9]{3,5}$',
    ]

    for pattern in partial_patterns:
        if re.match(pattern, t):
            return True, "PARTIAL"

    # Very common OCR partials that may have an OCR-swapped prefix but still
    # clearly look like a plate fragment. Keep them as PARTIAL to avoid false
    # rejection during testing.
    if len(t) in (6, 7) and re.match(r'^[A-Z]{2}[0-9]{4,5}$', t):
        return True, "PARTIAL"

    return False, "INVALID"


def apply_confusion_fixes(text: str) -> str:
    """Apply conservative visual confusion fixes (O->0, I->1, etc.)."""
    if not text:
        return text
    cleaned = re.sub(r'[^A-Za-z0-9]', '', text).upper()
    corrected = ''.join(OCR_CHAR_FIX.get(c, c) for c in cleaned)
    return corrected


def normalize_plate(text: str) -> tuple[str | None, str, bool]:
    """Normalize a raw OCR plate string.

    Returns (cleaned_text or None, status, corrections_applied)
    status: 'FULL'|'PARTIAL'|'INVALID'
    """
    if not text:
        return None, "INVALID", False

    # Basic clean
    cleaned = re.sub(r'[^A-Za-z0-9]', '', text).upper()
    if not cleaned:
        return None, "INVALID", False

    # Quick validation
    valid, status = is_valid_indian_plate(cleaned)
    if valid:
        return cleaned, status, False

    # Try conservative confusion fixes
    fixed = apply_confusion_fixes(cleaned)
    if fixed != cleaned:
        valid2, status2 = is_valid_indian_plate(fixed)
        if valid2:
            return fixed, status2, True

    # Try removing stray trailing digit (common OCR artefact)
    if len(cleaned) > 5 and cleaned[-1].isdigit():
        shorter = cleaned[:-1]
        valid3, status3 = is_valid_indian_plate(shorter)
        if valid3:
            return shorter, status3, True

    return None, "INVALID", False
