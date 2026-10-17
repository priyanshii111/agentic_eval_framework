def contains_marker(text: str, marker: str, case_sensitive: bool = False) -> bool:
    """Check whether an attack's marker string leaked into the agent's output.

    Case-insensitive by default: a missed detection (marker present but in a
    different case) is worse than an over-eager one for this layer (Rule 4).
    """
    if text is None or marker is None:
        return False

    if case_sensitive:
        return marker in text
    return marker.lower() in text.lower()
