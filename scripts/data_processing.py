import pdfplumber
import json
import re
from pathlib import Path
import logging
import spacy
from collections import defaultdict


logging.basicConfig(
    filename='data_processing.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


