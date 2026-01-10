import subprocess
from pathlib import Path
from urllib.parse import urlparse


def safe_filename_from_url(url: str) -> str:
    name = Path(urlparse(url).path).name or "document.pdf"
    return name.replace("%20", "_")


def download_pdf(url: str, dest_dir: Path, timeout: int = 30, max_size: int = 1_000_000) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    filename = safe_filename_from_url(url)
    dest = dest_dir / filename
    subprocess.run(
        ["curl", "-sSL", "--max-time", str(timeout), "-o", str(dest), url],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if dest.stat().st_size > max_size:
        dest.unlink(missing_ok=True)
        raise ValueError("Fichier téléchargé trop volumineux (>1Mo)")
    return dest
