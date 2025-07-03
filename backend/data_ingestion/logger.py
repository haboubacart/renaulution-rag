import logging
import os
from datetime import datetime

class MillisecondFormatter(logging.Formatter):
    def formatTime(self, record, datefmt=None):
        dt = datetime.fromtimestamp(record.created)
        if datefmt:
            return dt.strftime(datefmt)[:-3]  
        return dt.strftime("%H:%M:%S.%f")[:-3]

def get_logger(name: str = "reanult-rag") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Création du dossier logs si besoin
        os.makedirs("../logs", exist_ok=True)

        formatter = MillisecondFormatter("[%(asctime)s] %(levelname)s - %(message)s", "%H:%M:%S.%f")

        # Handler console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        # Handler fichier
        file_handler = logging.FileHandler("../logs/app.log", mode="a", encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
