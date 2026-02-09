from __future__ import annotations

def normalize_symbol(code: str) -> str:
    """
    Canonical symbol format used in our system: <code>.<EXCHANGE>
    e.g. 600519.SH, 000921.SZ, 688981.SH
    """
    code = str(code).strip()
    if not code.isdigit():
        raise ValueError(f"Invalid A-share code: {code}")
    if len(code) != 6:
        code = code.zfill(6)

    # Shanghai: 6xxxxx (includes 688xxx STAR, 689xxx)
    if code.startswith("6"):
        return f"{code}.SH"
    # Shenzhen: 0xxxxx, 2xxxxx, 3xxxxx (ChiNext)
    if code.startswith(("0", "2", "3")):
        return f"{code}.SZ"

    # Fallback (still keep it usable)
    return f"{code}.UNK"

def code_only(canonical: str) -> str:
    return canonical.split(".")[0]
