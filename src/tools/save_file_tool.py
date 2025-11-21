import os
import uuid

UPLOAD_DIR = "/tmp/data_whisperer_uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

def save_file_tool(file: str, filename: str) -> str:
    """
    Save an uploaded text file (for example CSV) and return a local file path.

    `file` is the file contents as a string from ADK.
    """
    file_id = str(uuid.uuid4())
    _, ext = os.path.splitext(filename)
    if not ext:
        ext = ".csv"

    file_path = os.path.join(UPLOAD_DIR, f"{file_id}{ext}")

    # Text mode, since ADK is giving us a string
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(file)

    return file_path
