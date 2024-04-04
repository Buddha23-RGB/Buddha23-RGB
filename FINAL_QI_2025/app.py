#%%
# !pip install flask-login
# !pip install Flask-Login
# !pip install Flask-Login
# #%%
# %pip install flask-login
# Import standard libraries
from flask import Flask
import os
import json
import sqlite3
from datetime import datetime

# Import third-party libraries
import pandas as pd
import yfinance as yf
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.style
import plotly.graph_objects as go
import plotly.io as plt_io
from dotenv import load_dotenv
from flask import Flask, render_template, request, url_for
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager
from werkzeug.security import generate_password_hash
from sqlalchemy.orm import Session, relationship
from sqlalchemy import create_engine, text, inspect

# Import local modules
from commons import *

# Load environment variables from .env file if it exists
load_dotenv(
    "C:\\Users\\joech\\OneDrive\\Documents\\Buddha23-RGB\\FINAL_QI_2025\\.github\\.env")

# Set up plotting styles
sns.set_style('whitegrid')
pd.core.common.is_list_like = pd.api.types.is_list_like
matplotlib.style.use('dark_background')
plt.style.use('dark_background')
plt_io.templates["custom_dark"] = plt_io.templates["plotly_dark"]
plt_io.templates["custom_dark"]['layout']['paper_bgcolor'] = '#30404D'
plt_io.templates["custom_dark"]['layout']['plot_bgcolor'] = '#30404D'
plt_io.templates['custom_dark']['layout']['yaxis']['gridcolor'] = '#4f687d'
plt_io.templates['custom_dark']['layout']['xaxis']['gridcolor'] = '#4f687d'

# Load config
with open('config.json') as f:
    config = json.load(f)

# Set up global variables
root_dir = config['root_dir']
rel_path = config['rel_path']
table_path = config['table_path']
benchmarks = config['benchmarks']
short_list = config['short_list']
windows = config['windows']
final_path = config['final_path']

# Initialize SQLAlchemy
db = SQLAlchemy()


def create_app():
    DB_FILE_PATH = os.path.join(root_dir, 'db', 'stock.db')
    DB_URI = f"sqlite:///{DB_FILE_PATH}"
    app = Flask(__name__)
    app.config['SQLALCHEMY_DATABASE_URI'] = DB_URI
    app.config['SECRET_KEY'] = os.getenv(
        'SECRET_KEY', 'your_default_secret_key')
    db.init_app(app)
    return app


def create_connection(db_file):
    try:
        conn = sqlite3.connect(db_file)
        return conn
    except Error as e:
        print(e)
        return None


def select_all_tasks(conn):
    cur = conn.cursor()
    cur.execute("SELECT * FROM tasks")
    rows = cur.fetchall()
    for row in rows:
        print(row)


app = create_app()
@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404


@app.errorhandler(500)
def internal_server_error(e):
    return render_template('500.html'), 500


@app.route('/disclaimer')
def disclaimer():
    return render_template('disclaimer.html')


@app.route('/privacy')
def privacy():
    return render_template('privacy.html')


@app.route('/contact')
def contact():
    return render_template('contact.html')


@app.route('/about')
def about():
    return render_template('about.html')


# @app.route('/chart')
# def chart():
#     return render_template('chart.html')



@app.route("/")
def index():
    from plotly.offline import plot
    portfolio = pd.read_csv(
        os.path.join(root_dir, 'db', 'final_tables', 'final_table.csv'), index_col=[0], parse_dates=True)
    portfolio = portfolio.iloc[-400:]
    # Create  trace for total_weightings
    trace1 = go.Scatter(
        x=portfolio.index,
        y=portfolio['MultiplierSum'],
        mode='lines',
        name='total_weightings'
    )
    layout = go.Layout(
        title='QI Custom Hourly Weightings Indicator',
        yaxis=dict(title='total_weightings'),
        yaxis2=dict(title='Multiplier'),
        plot_bgcolor='black',
        paper_bgcolor='black',
        width=1000,  # double the current width
        shapes=[
            # Line Horizontal
            dict(
                type="line",
                x0=portfolio.index.min(),
                y0=65,
                x1=portfolio.index.max(),
                y1=65,
                line=dict(
                    color="red",
                    width=2,
                    dash="dashdot",
                ),
            ),
            dict(
                type="line",
                x0=portfolio.index.min(),
                y0=0,
                x1=portfolio.index.max(),
                y1=0,
                line=dict(
                    color="white",
                    width=2,
                ),
            ),
            dict(
                type="line",
                x0=portfolio.index.min(),
                y0=-65,
                x1=portfolio.index.max(),
                y1=-65,
                line=dict(
                    color="green",
                    width=2,
                    dash="dashdot",
                )
            )
        ]
    )
    # Create a figure
    fig = go.Figure(data=[trace1], layout=layout)

    # Convert the figure to a div string
    div = plot(fig, output_type='div')
    weights = pd.read_csv(
        os.path.join(root_dir, 'db', 'weights_portfolio.csv'))
    return render_template('index.html', bearish=weights['Bearish Portfolio'][0], bullish=weights['Bullish Portfolio'][0], div=div)


if __name__ == '__main__':
    app.run()
#%%

#%%


def render_table(symbol):
    with open(f"templates/tables/{symbol}.html") as table:
        template = Template(table.read())
        # Load data for the table
        data = pd.read_csv(f'data/{symbol}.csv')
        return template.render(
            data=data.to_dict(orient='records')
        )


@app.route('/tables/<symbol>.html')
def table(symbol):
    return render_table(symbol)


@app.route('/data', methods=['POST', 'GET'])
def data():
    form_data = None
    if request.method == 'POST':
        form_data = request.form
        symbol = form_data.get('symbol')
        if symbol and isinstance(symbol, str):
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                # AJAX request, return only the table HTML
                return render_table(symbol)
            else:
                # Normal request, return the full page with the image
                image_url = url_for('static', filename=f'images/{symbol}.png')
                return render_template('/data.html', form_data=form_data, image_url=image_url)
        return render_template('index.html', form_data=form_data)
