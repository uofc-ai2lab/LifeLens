from datetime import datetime

def generate_export_filename(
    input_filename: str,
    service: str = "",
    ) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{timestamp}_{service}_{input_filename}.csv"
