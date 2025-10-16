import logging
import os
from datetime import datetime


def setup_logger(name: str = "rag", log_dir: str = "logs", level=logging.INFO) -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)

    log_date = datetime.now().strftime("%Y-%m-%d")
    log_filename = f"rag_{log_date}.log"
    log_path = os.path.join(log_dir, log_filename)

    log_files = sorted(
        [f for f in os.listdir(log_dir) if f.startswith("rag_") and f.endswith(".log")],
        reverse=True
    )
    for old_file in log_files[10:]:
        os.remove(os.path.join(log_dir, old_file))

    logger = logging.getLogger(name)
    logger.setLevel(level)

    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    if not logger.hasHandlers():
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger
