# redact.py
def redact_sensitive_info(data):
    redacted_data = data.copy()
    for column in redacted_data.columns:
        if "sensitive" in column.lower():
            redacted_data[column] = "[REDACTED]"
    return redacted_data
