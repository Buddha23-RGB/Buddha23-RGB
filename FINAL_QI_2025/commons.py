# %%
import array
import copy
import csv
import datetime
import datetime as dt
import json
import os
import warnings
from datetime import date
from distutils.version import LooseVersion
from urllib.parse import urljoin
import matplotlib

import matplotlib.cm as cm
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import plotly.io as plt_io
import python_utils
import pytz
import requests
import seaborn as sns
import yfinance as yf
from dotenv import load_dotenv
from flask import Flask, render_template, request
from IPython.display import HTML
from jinja2 import Template
from numpy.core.fromnumeric import squeeze
from pandas import Timestamp
from PIL import Image
from plotly import optional_imports
from pydantic import BaseModel
from scipy import integrate, stats
from setuptools import find_packages, setup
from sqlalchemy import (Boolean, Column, ForeignKey, Integer, Numeric, String,
                        create_engine)
from sqlalchemy.orm import Session, relationship
from dotenv import load_dotenv
import os
# !pip install schedule
# !pip install matplotlib
# Load environment variables from .env file if it exists
load_dotenv(
    "C:\\Users\\joech\\OneDrive\\Documents\\Buddha23-RGB\\FINAL_QI_2025\\.github\\.env")
import os
import json
import warnings
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import plotly.io as plt_io
import seaborn as sns

# os.chdir("rootdir=c:\\Users\\joech\\OneDrive\\Documents\\Buddha23-RGB\\FINAL_QI_2025")
# Set the style for seaborn and matplotlib
sns.set_style('whitegrid')
plt.style.use('dark_background')

rel_path = "C:\\Users\\joech\\OneDrive\\Documents\\Buddha23-RGB\\FINAL_QI_2025"


# Set custom dark theme for Plotly
plt_io.templates["custom_dark"] = plt_io.templates["plotly_dark"]
plt_io.templates["custom_dark"]['layout']['paper_bgcolor'] = '#30404D'
plt_io.templates["custom_dark"]['layout']['plot_bgcolor'] = '#30404D'
plt_io.templates['custom_dark']['layout']['yaxis']['gridcolor'] = '#4f687d'
plt_io.templates['custom_dark']['layout']['xaxis']['gridcolor'] = '#4f687d'

# Define commonly used lists
benchmarks = ['SPY', 'IWM', 'VXX', 'QQQ', 'SOXX', 'XLK', 'XRT', 'XLP', 'XLI', 'XLY', 'XLC', 'XLV',
              'XBI', 'XLU', 'IYT', 'GLD', 'GDX', 'SLV', 'UUP', 'USO', 'XLE', 'XLB', 'XLF', 'KBE', 'TLT',
              'XHB', 'ITB', 'RWR']

short_list = ['AAPL', 'TSLA', 'NVDA', 'SPY', 'SOXX', 'MSFT', 'GOOG', 'META', 'AMD', 'IWM', 'VXX', 'QQQ', 'XLK', 'XRT', 'XLP', 'XLI', 'XLY', 'XLC', 'XLV',
              'XBI', 'XLU', 'IYT', 'GLD', 'GDX', 'SLV', 'UUP', 'USO', 'XLE', 'XLB', 'XLF', 'KBE', 'TLT',
              'XHB', 'ITB', 'RWR']
windows = [15, 20, 25, 30, 40, 50,
           80, 90, 100, 125, 150, 180, 240]
table_path = "C:\\Users\\joech\\OneDrive\\Documents\\Buddha23-RGB\\FINAL_QI_2025\\db\\tables"
# Suppress warnings if needed
warnings.filterwarnings('ignore')

with open('config.json') as f:
    config = json.load(f)

root_dir = config['root_dir']
rel_path = config['rel_path']
table_path = config['table_path']
benchmarks = config['benchmarks']
short_list = config['short_list']
windows = config['windows']
#%%