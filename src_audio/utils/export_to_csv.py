import os, csv
from pathlib import Path
import pandas as pd
from datetime import datetime
from typing import List, Dict, Callable, Optional, Union
from src_audio.domain.constants import bcolors
from src_audio.utils.format_timestamp import format_timestamp
from src_audio.utils.generate_export_filename import generate_export_filename

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
    output_path: str,
    input_filename: str,
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

    full_output_name = generate_export_filename(input_filename, service)
    full_output_path=f"{output_path}/{full_output_name}"
    os.makedirs(os.path.dirname(full_output_path), exist_ok=True)
    
    
    if data is None or (isinstance(data, list) and len(data) == 0):
        if not empty_ok:
            print(bcolors.FAIL + "ERROR: No data to export." + bcolors.ENDC)
            return
        print(bcolors.WARNING + "WARNING: Exporting empty CSV." + bcolors.ENDC)
    
    try:
        # Case 1: DataFrame
        if isinstance(data, pd.DataFrame):
            if columns: 
                data = data[columns]
            data.to_csv(full_output_path, index=False)
            
        # Case 2: List[Dict]
        else:
            rows = _transform_row_fn(data, columns)
            
            if columns:
                rows = [{k: r.get(k,"") for k in columns} for r in rows]
            
            with open(full_output_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                
                if header:  writer.writerow(header)
                elif rows:  writer.writerow(rows[0].keys())
                
                for row in rows:    writer.writerow(row.values())
        print(bcolors.OKGREEN + f"CSV exported successfully -> {os.path.abspath(full_output_path)}"+bcolors.ENDC)
        
    except Exception as e:
        print(bcolors.FAIL + f"ERROR exporting CSV: {e}" + bcolors.ENDC)
        raise
