import os, csv
from pathlib import Path
import pandas as pd
from datetime import datetime
from typing import List, Dict, Callable, Optional, Union
from config.logger import Logger
from src_audio.utils.format_timestamp import format_timestamp

log = Logger("[audio][exporting]")

def _transform_row_fn(data, columns):
    # Apply default transformations
    transformed_rows = []
    for r in data:
        transformed_row = {}
        for k in (columns or r.keys()):
            val = r.get(k, "")
            if k == "start_time" or k == "end_time":
                try:
                    # Only format if numeric
                    if isinstance(val, (int, float)):
                        val = format_timestamp(val)
                    else:
                        # Already formatted → keep as-is
                        val = str(val)
                except Exception:
                    val = "00:00:00.000"
            elif k == "text":
                val = str(val).strip()
            elif k == "speaker":
                val = val
            transformed_row[k] = val
        transformed_rows.append(transformed_row)
    return transformed_rows


def export_to_csv(
    data: Union[List[Dict],pd.DataFrame],
    audio_chunk_path: Path,
    service: str = "",
    columns: Optional[List[str]] = None,
    header: Optional[List[str]] = None,
    empty_ok: bool = True,
):
    """
    Generic CSV exporter.

    Parameters:
    - data: list[dict] or pandas DataFrame
    - output_path: output file path
    - input_filename:  input file name (used in output file's name)
    - service: type of service exporting a csv
    - columns: ordered list of columns to include
    - header: optional custom header (for csv.writer style)
    - _transform_row_fn: optional function to transform each row
    - empty_ok: whether to write empty CSV if data is empty
    """
    
    output_filename = f"{service}_{audio_chunk_path.parent.stem}.csv"
    full_output_path = audio_chunk_path.parent / output_filename
    full_output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Determine columns for empty CSV
    if columns:
        cols_to_write = columns
    elif isinstance(data, pd.DataFrame):
        cols_to_write = data.columns.tolist()
    elif isinstance(data, list) and len(data) > 0:
        cols_to_write = list(data[0].keys())
    else:
        # No data → fallback to header parameter or empty list
        cols_to_write = header or []
    
    # Warn if data is empty
    if not data or (isinstance(data, list) and len(data) == 0) or (isinstance(data, pd.DataFrame) and data.empty):
        if not empty_ok:
            log.error("No data to export")
            return
        log.warning(f"Exporting empty CSV → {full_output_path}")
    
    try:
        # Case 1: DataFrame
        if isinstance(data, pd.DataFrame):
            # Ensure all requested columns exist
            if columns: 
                for col in columns:
                    if col not in data:
                        data[col] = None
                data = data[columns]
            data.to_csv(full_output_path, index=False)
            
        # Case 2: List[Dict]
        else:
            rows = _transform_row_fn(data, columns)
            # Ensure all columns exist for empty rows
            for r in rows:
                for col in cols_to_write:
                    if col not in r:
                        r[col] = ""
            
            with full_output_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(cols_to_write)  # always write header
                for row in rows:
                    writer.writerow([row.get(c, "") for c in cols_to_write])
                
        log.success(f"CSV exported -> {full_output_path.resolve()}")
        
    except Exception as e:
        log.error(f"Error exporting CSV: {e}")
        raise

    return full_output_path