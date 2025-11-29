import os
import uuid
from typing import Any, Dict

from ..utils.errors import FILE_IO_ERROR, exception_to_error, wrap_success
from ..utils.paths import get_artifact_path

UPLOAD_DIR = get_artifact_path("data_whisperer_uploads", create_dir=True)


def save_file_tool(file: str, filename: str) -> Dict[str, Any]:
    """
    Save an uploaded text file (for example CSV) and return a local file path.

    `file` is the file contents as a string from ADK.
    """
    try:
        file_id = str(uuid.uuid4())
        _, ext = os.path.splitext(filename)
        if not ext:
            ext = ".csv"

        file_path = os.path.join(UPLOAD_DIR, f"{file_id}{ext}")

        # Text mode, since ADK is giving us a string
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(file)

        return wrap_success({"file_path": file_path})
    except (IOError, OSError, PermissionError) as e:
        return exception_to_error(
            FILE_IO_ERROR,
            e,
            hint="Check file system permissions and available disk space",
        )
