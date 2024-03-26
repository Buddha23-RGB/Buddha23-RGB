#%%
!pip install flask-login
!pip install flask_login
from flask import Flask, render_template, request, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
import pandas as pd
import json
import flask
from flask import Flask,request,render_template
from package import fetch
import commons 
from commons import *
from datetime import datetime
from sqlalchemy import create_engine
import plotly.offline as pyo
import plotly.express as px
from distutils.version import LooseVersion
from plotly import optional_imports
import plotly.io as pio
import os
import yfinance as yf
from flask import Flask, render_template, request
from IPython.display import HTML
from jinja2 import Template
from numpy.core.fromnumeric import squeeze
from PIL import Image
from pydantic import BaseModel
from scipy import integrate, stats
from setuptools import find_packages, setup
from sqlalchemy import (Boolean, Column, ForeignKey, Integer, Numeric, String,
                        create_engine)
from sqlalchemy.orm import Session, relationship
import copy
import csv
import datetime
import datetime as dt
import json
import os
import warnings
from datetime import date
from urllib.parse import urljoin
import matplotlib
import matplotlib.cm as cm
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
import plotly.io as plt_io
import python_utils
import plotly.graph_objects as go
from sqlalchemy import inspect
from sqlalchemy import create_engine, text
import pandas as pd
sns.set_style('whitegrid')

pd.core.common.is_list_like = pd.api.types.is_list_like

style = matplotlib.style.use('dark_background')
plt.style.use('dark_background')
plt_io.templates["custom_dark"] = plt_io.templates["plotly_dark"]

plt_io.templates["custom_dark"]['layout']['paper_bgcolor'] = '#30404D'
plt_io.templates["custom_dark"]['layout']['plot_bgcolor'] = '#30404D'

plt_io.templates['custom_dark']['layout']['yaxis']['gridcolor'] = '#4f687d'
plt_io.templates['custom_dark']['layout']['xaxis']['gridcolor'] = '#4f687d'

windows = [15, 20, 25, 30, 40, 50, 80, 90, 100, 125, 150, 180, 240]


# Load config
with open('config.json') as f:
    config = json.load(f)

root_dir = config['root_dir']
rel_path = config['rel_path']
table_path = config['table_path']
benchmarks = config['benchmarks']
short_list = config['short_list']
windows = config['windows']

DB_FILE_PATH = "C:/Users/joech/OneDrive/Documents/Buddha23-RGB/FINAL_QI_2025/db/stock.db"
DB_URI = f"sqlite:///{DB_FILE_PATH}"

# Initialize Flask app
app = Flask(__name__)

# Initialize SQLAlchemy
app.config['SQLALCHEMY_DATABASE_URI'] = DB_URI
db = SQLAlchemy(app)

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        symbol = request.form.get('symbol')
        if not symbol:
            flash('Symbol is required.')
            return redirect(url_for('home'))

        # Now you can use the symbol variable in your code
        # ...

        # Assuming you have the paths to the images in the variables price_chart and trending_chart
        price_chart = pd.read_csv(
            f'C:/Users/joech/OneDrive/Documents/Buddha23-RGB/FINAL_QI_2025/db/tables/{symbol}.csv')
        trending_chart = pd.read_csv(
            f'C:/Users/joech/OneDrive/Documents/Buddha23-RGB/FINAL_QI_2025/db/tables/{symbol}_divergence.csv')

        return render_template('index.html', tables=[price_chart.to_html(classes='data', header="true"), trending_chart.to_html(classes='data', header='true')])

    return render_template('index.html')

@app.route("/",methods=["GET","POST"])
def index():
    #info={}
    if request.method=="POST":
        ticker=request.form["ticker"]
        info=fetch.company_info(ticker)
        return render_template("home.html",info=info)
    return render_template("home.html")
if __name__ == '__main__':
    app.run()
#%%



