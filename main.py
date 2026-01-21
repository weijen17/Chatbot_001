"""
Main entry point for the Chatbot
"""

import os
from dotenv import load_dotenv
import argparse
from frontend import streamlit_run
import pandas as pd

from src.config.settings import settings
from src.agents import Chatbot


if __name__ == "__main__":
    init_chatbot = Chatbot()
    streamlit_run(init_chatbot)



