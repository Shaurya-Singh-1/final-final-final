def parse_bool(value):
    cleaned = value.lower()
    if cleaned == "true":
        return True
    if cleaned == "false":
        return False
    raise ValueError(f"unsupported boolean value: {value}")
