import os
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

ALPHA_VANTAGE_API_KEY: Optional[str] = os.getenv("ALPHA_VANTAGE_API_KEY")

FINBERT_MODEL_NAME: str = "ProsusAI/finbert"