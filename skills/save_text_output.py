import json, os, re
from datetime import datetime
from pydantic import BaseModel, Field
from typing import Annotated

class TextOutputFields(BaseModel):
    filename: Annotated[str, Field(description="The output filename that we need to write to")]
    content: Annotated[str, Field(description="The message received by the agents that needs to be written to a file")]
    ext: str = ".txt"
    output_dir: str = "output"


def save_text_output(input: TextOutputFields) -> None:
    os.makedirs(input.output_dir, exist_ok=True)
    file_path = os.path.join("output", input.filename)
    with open(file_path, 'a') as f:
        f.write(input.content)
    print(f"Saved code to {file_path}")