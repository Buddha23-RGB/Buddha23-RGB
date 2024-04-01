#%%
from flask import Flask, render_template, request, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
import pandas as pd
import json
import flask
from flask import Flask, request, render_template
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
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, login_user, logout_user, login_required
from werkzeug.security import generate_password_hash
from datetime import datetime
import json
import os
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


# Load config
with open('config.json') as f:
    config = json.load(f)

DB_FILE_PATH = "C:/Users/joech/OneDrive/Documents/Buddha23-RGB/FINAL_QI_2025/db/stock.db"
DB_URI = f"sqlite:///{DB_FILE_PATH}"

# Initialize Flask app
app = Flask(__name__)

# Initialize SQLAlchemy
db = SQLAlchemy(app)
login_manager = LoginManager()

def create_app():
    app = Flask(__name__)
    app.config['SQLALCHEMY_DATABASE_URI'] = DB_URI
    app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'your_default_secret_key')

    db.init_app(app)
    login_manager.init_app(app)
    login_manager.login_view = 'login'

    return app

@login_manager.user_loader
def load_user(user_id):
    from models import User  # Import here to avoid circular dependency
    return User.query.get(int(user_id))

app = create_app()

@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_server_error(e):
    return render_template('500.html'), 500

@app.route("/register", methods=['GET', 'POST'])
def register():
    # Your registration logic here

@app.route("/login", methods=['GET', 'POST'])
def login():
    # Your login logic here

@app.route("/logout")
@login_required
def logout():
    logout_user()
    flash('You have been logged out.', 'success')
    return redirect(url_for('homepage'))

@app.route('/disclaimer')
def disclaimer():
    return render_template('disclaimer.html')

@app.route('/privacy')
def privacy():
    return render_template('privacy.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/about', methods=['GET'])
def about():
    return render_template('about.html')

@app.route('/chart')
def chart():
    return render_template('chart.html')


class Transaction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.String, db.ForeignKey('user.id'))
    symbol = db.Column(db.String)
    shares = db.Column(db.Integer)
    price = db.Column(db.Float)
    transaction_type = db.Column(db.String)
    date = db.Column(db.DateTime)
    multiplier = db.Column(db.Float)

    user = db.relationship('User', backref='transactions')

@app.route('/portfolio', methods=['GET', 'POST'])
@login_required
def portfolio():
    form = TransactionForm()
    if form.validate_on_submit():
        transaction = Transaction(
            user_id=current_user.id,
            symbol=form.symbol.data,
            shares=form.shares.data,
            price=form.price.data,
            transaction_type=form.transaction_type.data,
            multiplier=form.multiplier.data
        )
        db.session.add(transaction)
        db.session.commit()
        flash('Transaction added!')
        return redirect(url_for('portfolio'))
    return render_template('portfolio.html', form=form)

@app.route('/portfolio/<username>', methods=['GET', 'POST'])
def user_portfolio(username):
    form = PortfolioForm()
    if form.validate_on_submit():
        # Handle the form submission
        # Assuming you have a User model with a username field
        user = User.query.filter_by(username=username).first()
        if user:
            # Assuming you want to add a new transaction for this user
            transaction = Transaction(
                user_id=user.id,
                symbol=form.symbol.data,
                shares=form.shares.data,
                price=form.price.data,
                transaction_type=form.transaction_type.data,
                multiplier=form.multiplier.data
            )
            db.session.add(transaction)
            db.session.commit()
            flash('Transaction added for user!')
        else:
            flash('User not found!')
        return redirect(url_for('user_portfolio', username=username))
    return render_template('user_portfolio.html', form=form)

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


@app.route("/")
def index():
    from plotly.offline import plot
    portfolio = pd.read_csv(
        "C:/Users/joech/source/repos/quantinvests_2025/app/db/portfolio_tables/total_weightings.csv", index_col=[0], parse_dates=True)
    portfolio = portfolio.iloc[-400:]
    # Create  trace for total_weightings
    trace1 = go.Scatter(
        x=portfolio.index,
        y=portfolio['total_weightings'],
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
        "C:/Users/joech/source/repos/quantinvests_2025/app/db/weights_portfolio.csv")
    return render_template('index.html', bearish=weights['Bearish Portfolio'][0], bullish=weights['Bullish Portfolio'][0], div=div)


@app.route('/submit', methods=['POST'])
def submit():
    # Extract the form data
    starting_capital = request.form.get('starting-capital')
    risk_tolerance = request.form.get('risk-tolerance')
    investment_horizon = request.form.get('investment-horizon')

    # Perform some calculations or processing here
    # For example, generate investment recommendations based on the input parameters
    # This is a placeholder for the actual logic
    recommendations = {
        'message': 'Investment recommendations based on your input will be displayed here.',
        'starting_capital': starting_capital,
        'risk_tolerance': risk_tolerance,
        'investment_horizon': investment_horizon
    }

    # Instead of returning JSON, store the recommendations in the session and redirect to the portfolio page
    # You can also use flash messages to display the 'message'
    flash(recommendations['message'])
    return redirect(url_for('portfolio', recommendations=recommendations))


# @app.route('/portfolio')
# def portfolio():
#     # Retrieve recommendations from the session or use defaults
#     recommendations = request.args.get('recommendations', {})
#     return render_template('UI.html', recommendations=recommendations)

if __name__ == '__main__':
    app.run()
#%%