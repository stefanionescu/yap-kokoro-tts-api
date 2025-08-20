import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from logger import setup_logger
setup_logger()

from app import create_app

# Create the FastAPI application
app = create_app()