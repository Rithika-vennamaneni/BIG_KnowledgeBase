import os
from dotenv import load_dotenv

load_dotenv()

# Azure Storage
AZURE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
CONTAINER_NAME = "healthcare-policies"

# Policy Settings
MIN_CONFIDENCE_THRESHOLD = 0.7  # Flag policies below this
SIMILARITY_THRESHOLD = 0.85     # For finding duplicate policies
DEFAULT_POLICY_TTL_DAYS = 730   # Consider policies older than 2 years as expired

# Anthropic (optional)
#ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")  # For LLM-based extraction

# Directories
TEMP_PDF_DIR = "./temp_pdfs"
TEMP_JSON_DIR = "./temp_json"