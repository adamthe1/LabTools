"""
Central .env loader with absolute path - works from any directory.
"""
import os
from dotenv import load_dotenv

# Load .env from project root with absolute path
_PROJECT_ROOT = '/home/adamgab/PycharmProjects/GaitPredict'
_ENV_PATH = os.path.join(_PROJECT_ROOT, '.env')

load_dotenv(_ENV_PATH, interpolate=True)

