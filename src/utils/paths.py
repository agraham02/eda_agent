"""
Cross-platform path handling utilities.

Provides helper functions for working with file paths across different operating systems
(Windows, macOS, Linux) to ensure the application works consistently everywhere.
"""

import os
import tempfile
from pathlib import Path
from typing import Optional


def get_temp_dir() -> str:
    """
    Get the system's temporary directory in a cross-platform way.

    Returns the appropriate temp directory for the current OS:
    - Windows: C:\\Users\\<user>\\AppData\\Local\\Temp
    - macOS: /var/folders/...
    - Linux: /tmp

    Returns:
        str: Absolute path to the system temporary directory
    """
    return tempfile.gettempdir()


def get_artifact_path(
    subdirectory: str, filename: Optional[str] = None, create_dir: bool = True
) -> str:
    """
    Get a path for storing application artifacts (plots, uploads, etc.).

    Creates a subdirectory within the system temp directory for organizing
    artifacts. Optionally creates the directory if it doesn't exist.

    Args:
        subdirectory: Name of subdirectory within temp (e.g., "data_whisperer_plots")
        filename: Optional filename to append to the path
        create_dir: If True, create the directory if it doesn't exist (default: True)

    Returns:
        str: Absolute path to the artifact location

    Examples:
        >>> get_artifact_path("plots")
        '/tmp/plots'  # or C:\\Users\\...\\Temp\\plots on Windows

        >>> get_artifact_path("plots", "chart.png")
        '/tmp/plots/chart.png'
    """
    temp_dir = get_temp_dir()
    artifact_dir = os.path.join(temp_dir, subdirectory)

    if create_dir:
        os.makedirs(artifact_dir, exist_ok=True)

    if filename:
        return os.path.join(artifact_dir, filename)

    return artifact_dir


def ensure_dir_exists(path: str) -> None:
    """
    Ensure a directory exists, creating it if necessary.

    Args:
        path: Path to directory to create
    """
    os.makedirs(path, exist_ok=True)


def normalize_path(path: str) -> str:
    """
    Normalize a file path for the current operating system.

    Converts path separators and resolves relative paths to absolute paths.

    Args:
        path: Path to normalize

    Returns:
        str: Normalized absolute path
    """
    return os.path.abspath(os.path.normpath(path))
