from datetime import datetime

def generate_export_filename(service_name: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{timestamp}_{service_name}"
