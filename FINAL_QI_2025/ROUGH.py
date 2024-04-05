#%%
!pip install flask-login

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




pivot_table = df['Multiplier'].unstack(level='Ticker')
pivot_table.fillna(0, inplace=True)

signal_pivot = df['Signal'].unstack(level='Ticker')
multiplier_sum = pivot_table.sum(axis=1)
signal_sum = signal_pivot.sum(axis=1)
# %%

portfolio = pd.DataFrame(multiplier_sum, columns=['Multiplier'])
# %%
portfolio['abs_total_weightings'] = total_weight[::-1].sum(axis=1)
# Create a line plot of total_weightings
fig = go.Figure()
fig.update_layout(
    autosize=False,
    width=1200,
    height=600,
)
fig.add_trace(go.Scatter(x=portfolio.index,
              y=portfolio['total_weightings'], mode='lines', name='total_weightings'))

# Add horizontal lines
fig.add_shape(type="line", x0=portfolio.index.min(), x1=portfolio.index.max(
), y0=70, y1=70, line=dict(color="Red", width=1, dash="dash"))
fig.add_shape(type="line", x0=portfolio.index.min(
), x1=portfolio.index.max(), y0=0, y1=0, line=dict(color="White", width=1))
fig.add_shape(type="line", x0=portfolio.index.min(), x1=portfolio.index.max(
), y0=-70, y1=-70, line=dict(color="Green", width=1, dash="dash"))

# Apply the custom dark theme
fig.update_layout(template='custom_dark',
                  title="QI Custom Hourly Weightings Indicator")

# Save the figure as an HTML file
pio.write_html(
    fig, 'C:/Users/joech/source/repos/quantinvests_2025/app/templates/charts/total_weightings_indicator.html')

fig.show()
portfolio['total_weightings'] = portfolio.sum(axis=1)

total_weight = abs(portfolio)
total_weight


//anchor()







# @app.route("/register", methods=['GET', 'POST'])
# def register():
#     # Your registration logic here

# @app.route("/login", methods=['GET', 'POST'])
# def login():
#     # Your login logic here

# @app.route("/logout")
# @login_required
# def logout():
#     logout_user()
#     flash('You have been logged out.', 'success')
#     return redirect(url_for('homepage'))



# login_manager.init_app(app)


# @login_manager.user_loader
# def load_user(user_id):
#     from models import User  # Import here to avoid circular dependency
#     return User.query.get(int(user_id))







# @app.route('/submit', methods=['POST'])
# def submit():
#     # Extract the form data
#     starting_capital = request.form.get('starting-capital')
#     risk_tolerance = request.form.get('risk-tolerance')
#     investment_horizon = request.form.get('investment-horizon')

#     # Perform some calculations or processing here
#     # For example, generate investment recommendations based on the input parameters
#     # This is a placeholder for the actual logic
#     recommendations = {
#         'message': 'Investment recommendations based on your input will be displayed here.',
#         'starting_capital': starting_capital,
#         'risk_tolerance': risk_tolerance,
#         'investment_horizon': investment_horizon
#     }

#     # Instead of returning JSON, store the recommendations in the session and redirect to the portfolio page
#     # You can also use flash messages to display the 'message'
#     flash(recommendations['message'])
#     return redirect(url_for('portfolio', recommendations=recommendations))


# @app.route('/portfolio')
# def portfolio():
#     # Retrieve recommendations from the session or use defaults
#     recommendations = request.args.get('recommendations', {})
#     return render_template('UI.html', recommendations=recommendations)

use the multiplier table instead of df
#%%


# %%
portfolio.to_csv(
    "C:/Users/joech/source/repos/quantinvests_2025/app/db/portfolio_tables/total_weightings.csv")
portfolio = portfolio[-400:]
ax = portfolio['total_weightings'].plot(figsize=(14, 6))
ax.axhline(65, color='r', linestyle='--')  # Add horizontal line at y=70
ax.axhline(0, color='w', linestyle='-')  # Add horizontal line at y=70
ax.axhline(-65, color='g', linestyle='--')  # Add horizontal line at y=-70
fig2 = portfolio['total_weightings'].plot(
    ax=ax, secondary_y='Multiplier', style='--', title="QI Custom Hourly Weightings Indicator")
ax.figure.savefig(
    "C:/Users/joech/source/repos/quantinvests_2025/app/static/total_weightings_indicator.jpg")
# portfolio = pd.read_csv("/workspaces/quantinvests_2025/app/db/multiplier_data.csv")
df
#%%


now = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
df.to_csv(f'C:/Users/joech/OneDrive/Documents/Buddha23-RGB/FINAL_QI_2025/db/user_portfolio/user_portfolio_{now}.csv')
#%%
columns_to_drop = ['Sig_div', 'Sig_ds',
                   'Sig_cor', 'Sig_ci', 'Sig_idx']
df




#%%
df.drop(columns=columns_to_drop, inplace=True)

# Apply the color_negative_red function to style the DataFrame
styled_df = df.reset_index(drop=False).style.applymap(
    color_negative_red, subset=['Multiplier', 'Mult_change', 'shares'])
styled_df.to_html("/templates/tables/table_css.html")
styled_df
#%%

fig = go.Figure(data=[go.Pie(labels=mult_bull.index, values=mult_bull.Multiplier,
                textinfo='label+percent', insidetextorientation='radial')])
fig.update_layout(title_text='Bullish Portfolio',
                  template='plotly_dark', font=dict(size=20))
pyo.plot(fig, filename='C:/Users/joech/source/repos/quantinvests_2025/app/static/pie_table_long.html', auto_open=True)
fig.show()
#%%
fig2 = go.Figure(data=[go.Pie(labels=mult_bear.index, values=mult_bear.Multiplier,
                 textinfo='label+percent', insidetextorientation='radial')])
fig2.update_layout(title_text='Bearish Portfolio',
                   template='plotly_dark', font=dict(size=20))
pyo.plot(fig2, filename='C:/Users/joech/source/repos/quantinvests_2025/app/static/pie_table_short.html', auto_open=True)
fig2.show()
#%%

print(f"Bullish Portfolio: {wb}%")
print(f"Bearish Portfolio: {wp}%")

# %%
dfs = {}

symbols = ss.short_list
for symbol in symbols:

    df = pd.read_csv(
        f"C:/Users/joech/source/repos/quantinvests_2025/app/db/portfolio_tables/{symbol}.csv", index_col='Datetime')
    dfs[symbol] = pd.concat([df])

# %%
portfolio = pd.DataFrame()
for symbol in symbols:
    portfolio[symbol] = dfs[symbol]["Multiplier"].replace(
        0, pd.NA).fillna(method='ffill')

# %%

#%%

def get_daily_data(tickers):

    def data(ticker):
        return pd.read_csv(f"C:/Users/joech/source/repos/quantinvests_2025/app/db/portfolio_tables/{ticker}.csv", index_col='Datetime')
    datas = map(data, tickers)
    return (pd.concat(datas, keys=tickers, names=['Ticker', 'Datetime']))


symbols = ss.short_list
db = get_daily_data(symbols)
db.round(2)
db.to_csv(
    "C:/Users/joech/source/repos/quantinvests_2025/app/db/portfolio_tables/final_signal_table.csv")

db.replace(0, pd.NA, inplace=True)
db.fillna(method='ffill', inplace=True)
db.to_sql("signal_table", engine, if_exists="replace")

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


app = Flask(__name__)



# %%
portfolio# %%
from flask import send_from_directory
from flask import Flask, flash, redirect, render_template, request, url_for, Blueprint
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from werkzeug.security import check_password_hash, generate_password_hash
from config import DevelopmentConfig, ProductionConfig
from forms import RegistrationForm, LoginForm, TransactionForm
from models import User, Transaction
from jinja2 import Template
from portfolio import portfolio_bp
import os
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
sns.set_style('whitegrid')

pd.core.common.is_list_like = pd.api.types.is_list_like

# this helps us get the theme settings
# Necessary imports:
# this is for simple plotting with plotly express
style = matplotlib.style.use('dark_background')
plt.style.use('dark_background')
# create our custom_dark theme from the plotly_dark template
plt_io.templates["custom_dark"] = plt_io.templates["plotly_dark"]

# set the paper_bgcolor and the plot_bgcolor to a new color
plt_io.templates["custom_dark"]['layout']['paper_bgcolor'] = '#30404D'
plt_io.templates["custom_dark"]['layout']['plot_bgcolor'] = '#30404D'

# you may also want to change gridline colors if you are modifying background
plt_io.templates['custom_dark']['layout']['yaxis']['gridcolor'] = '#4f687d'
plt_io.templates['custom_dark']['layout']['xaxis']['gridcolor'] = '#4f687d'
# app.py


# Initialize db and login_manager as global variables
db = SQLAlchemy()
login_manager = LoginManager()


def create_app():
    # Initialize Flask app
    app = Flask(__name__)
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///C:/Users/joech/source/repos/quantinvests_2025/app/db/portfolio.db'
    # Add a secret key for CSRF protection
    app.config['SECRET_KEY'] = 'your_secret_key'

    # Initialize db with app
    db.init_app(app)

    # Initialize Flask-Login with app
    login_manager.init_app(app)

    # Specify the login view
    login_manager.login_view = 'login'

    # Initialize Flask-Migrate with app and db
    migrate = Migrate(app, db)

    # Register the portfolio Blueprint
    app.register_blueprint(portfolio_bp, url_prefix='/portfolio')

    return app

# Define the user_loader callback for Flask-Login


@login_manager.user_loader
def load_user(user_id):
    from models import User  # Import here to avoid circular dependency
    return User.query.get(int(user_id))


# Create an instance of the Flask application
app = create_app()


@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404


@app.errorhandler(500)
def internal_server_error(e):
    return render_template('500.html'), 500


@app.route("/register", methods=['GET', 'POST'])
def register():
    form = RegistrationForm()
    if form.validate_on_submit():
        hashed_password = generate_password_hash(form.password.data)
        user = User(username=form.username.data, password_hash=hashed_password,
                    starting_capital=form.starting_capital.data, capital=form.starting_capital.data)
        db.session.add(user)
        db.session.commit()
        flash('Your account has been created! You are now able to log in', 'success')
        return redirect(url_for('login'))
    return render_template('register.html', title='Register', form=form)


@app.route("/login", methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        if user and user.check_password(form.password.data):
            login_user(user)
            flash('Logged in successfully.', 'success')
            return redirect(url_for('homepage'))
        else:
            flash('Invalid username or password.', 'danger')
    return render_template('login.html', form=form)


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

# class Transaction(db.Model):
#     id = db.Column(db.Integer, primary_key=True)
#     user_id = db.Column(db.String, db.ForeignKey('user.id'))
#     symbol = db.Column(db.String)
#     shares = db.Column(db.Integer)
#     price = db.Column(db.Float)
#     transaction_type = db.Column(db.String)
#     date = db.Column(db.DateTime)
#     multiplier = db.Column(db.Float)


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
        pass
    return render_template('user_portfolio.html', form=form)


def render_table(symbol):
    with open(f"templates/tables/{symbol}.html") as table:
        template = Template(table.read())
        return template.render(
            data=data
        )


@app.route('/tables/<symbol>.html')
def table(symbol):
    return render_template(f'/tables/{symbol}.html')


@app.route('/data', methods=['POST', 'GET'])
def data():
    form_data = None
    if request.method == 'POST':
        form_data = request.form
        symbol = form_data.get('symbol')
        if symbol and isinstance(symbol, str):
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                # AJAX request, return only the table HTML
                return render_template(f'/tables/{symbol}.html')
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
    
from flask import Flask, session, g
from flask import render_template, json, redirect, url_for, request, flash
from flask_security import Security, SQLAlchemyUserDatastore
from flask_security import UserMixin, RoleMixin, login_required, current_user
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.ext.indexable import index_property
from flask_mail import Mail
from decimal import *
import os
import requests
import datetime
"""
# Setup flask_mail. This is used to email users
# Can send a welcome email on registration or forgotten password link
"""
mail = Mail()
MAIL_SERVER = 'smtp.gmail.com'
MAIL_PORT = 465
MAIL_USE_TLS = False
MAIL_USE_SSL = True
MAIL_USERNAME = 'ryantest216@gmail.com'
MAIL_PASSWORD = '99Google99'
app = Flask(__name__)  # Setup flask app
app.config.from_object(__name__)  # Setup app config
mail.init_app(app)  # Initialise flask_mail with this app
"""
# My Config settings for flask security and sqlachemy. 
# Debug is left set to false. Set to true for live reload and debugging
"""
app.config['DEBUG'] = False  # Disable this when ready for production
app.config['SECRET_KEY'] = 'super-secret'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///data/database.db'
# This enables the register option for flask_security
app.config['SECURITY_REGISTERABLE'] = True
# This enables the forgot password option for flask_security
app.config['SECURITY_RECOVERABLE'] = True
app.config['SECURITY_POST_LOGIN_VIEW'] = 'dashboard'
app.config['SECURITY_POST_REGISTER_VIEW'] = 'dashboard'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = True

db = SQLAlchemy(app)  # Create database connection object with SQLAlchemy

"""
# Models for Database.
"""
roles_users = db.Table('roles_users',
                       db.Column('user_id', db.Integer(),
                                 db.ForeignKey('user.id')),
                       db.Column('role_id', db.Integer(), db.ForeignKey('role.id')))

users_currencies = db.Table('users_currencies',
                            db.Column('user_id', db.Integer(),
                                      db.ForeignKey('user.id')),
                            db.Column('amount', db.Integer()),
                            db.Column('ticker', db.String(255)),
                            db.Column('last', db.Float()),
                            db.Column('bid', db.Float()),
                            db.Column('ask', db.Float())
                            )
# This class is used to model the table which will hold Users
# Contains a backreference to the Role class for User/Admin role possiblities


class Role(db.Model, RoleMixin):
    __tablename__ = "role"
    id = db.Column(db.Integer(), primary_key=True)
    name = db.Column(db.String(80), unique=True)
    description = db.Column(db.String(255))

# This class is used to model the table which will hold Users
# Contains a backreference to the Role class for User/Admin role possiblities


class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(255), unique=True)
    password = db.Column(db.String(255))
    active = db.Column(db.Boolean())
    confirmed_at = db.Column(db.DateTime())
    roles = db.relationship('Role', secondary=roles_users,
                            backref=db.backref('users', lazy='dynamic'))
# This class is used to model the table which will hold the currencies themselves
# Information acquired via the /GET/ method of a publicly available REST API


class Currency(db.Model, UserMixin):
    __tablename__ = "Currency"
    id = db.Column(db.Integer, primary_key=True)
    ticker = db.Column(db.String(255), unique=True)
    last = db.Column(db.String(255))
    ask = db.Column(db.String(255))
    bid = db.Column(db.String(255))
    timestamp = db.Column(db.DateTime())


# This class is used to model the table which will hold each users currency
# Contains id as a foreign key from User


class UserCurrency(db.Model, UserMixin):
    __tablename__ = "users_cur"
    trans_id = db.Column(db.Integer, primary_key=True, index=True)
    id = db.Column(db.Integer)

    amount = db.Column(db.Numeric())
    ticker = db.Column(db.String(255))
    priceInBTC = db.Column(db.Numeric())
    priceInUSD = db.Column(db.Numeric())
    priceInEUR = db.Column(db.Numeric())
    priceInCHY = db.Column(db.Numeric())
    last = db.Column(db.String(255))
    ask = db.Column(db.String(255))
    bid = db.Column(db.String(255))
    timestamp = db.Column(db.DateTime())
    index = index_property('id', 'index')


class Stock(db.Model, UserMixin):
    __tablename__ = "Stocks"
    id = db.Column(db.Integer, primary_key=True)
    ticker = db.Column(db.String(255), unique=True)
    last = db.Column(db.String(255))
    market = db.Column(db.String(255))
    timestamp = db.Column(db.DateTime())


# This class is used to model the table which will hold each users stock investments
# Contains id as a foreign key from User


class UserStocks(db.Model, UserMixin):
    __tablename__ = "users_stocks"
    trans_id = db.Column(db.Integer, primary_key=True, index=True)
    id = db.Column(db.Integer)

    amount = db.Column(db.Numeric())
    ticker = db.Column(db.String(255))
    market = db.Column(db.String(255))
    priceInBTC = db.Column(db.Numeric())
    priceInUSD = db.Column(db.Numeric())
    priceInEUR = db.Column(db.Numeric())
    priceInCHY = db.Column(db.Numeric())
    last = db.Column(db.String(255))
    timestamp = db.Column(db.DateTime())
    index = index_property('id', 'index')


# Setup user_datastore and sqlalchemy for flask_security to use
user_datastore = SQLAlchemyUserDatastore(db, User, Currency)
security = Security(app, user_datastore)


# Create a user to test with
@app.before_first_request
def create_user():
    # Possible implementation
    # Query db for users by email
    # if dummy user does not exist, create him and attempt to fill the database
    # if not perhaps check the db and if no currencies are there fill that up too.
    if db is None or User.query.first() is None:
        print("No Users found, creating test user")
        db.create_all()
        user_datastore.create_user(
            email='ryan@gordon.com', password='password', confirmed_at=datetime.datetime.now())
        r = requests.get('https://poloniex.com/public?command=returnTicker')
        # Pull JSON market data from Bittrex
        b = requests.get(
            'https://bittrex.com/api/v1.1/public/getmarketsummaries')
        # Print value to user and assign to variable
        data = r.json()
        # Print value to user and assign to variable
        bittrex = b.json()

        for key in data.keys():
            u = Currency(ticker=key, last=data[key]['last'], ask=data[key]['lowestAsk'],
                         bid=data[key]['highestBid'], timestamp=datetime.datetime.now())
            db.session.add(u)

        db.session.commit()
    else:
        print("Found Users in DB")


"""
Views/ Routes for the webapp. homepage, login and register have their own pages.  
All other pages inherit from the index.html page which holds the UI for the webapp (menu and nav)
This is done using Jinja2 Syntaxing Engine. Designed by the Flask team, pocoo

"""
# The default route. Provides a landing page with info about the app and options to login/register


@app.route('/')
def landing_page():
    db.create_all()
    return render_template("homepage.html")
# This route provides a basic UI view of the app with no content. Will be removed in production


@app.route('/index')
@login_required
def index():
    return render_template("index.html")
# All this does is log out the user if any and


@app.route('/logout')
def logout():
    logout_user(self)

# This route is the main starter view of the app and contains info from the other sections


@app.route('/dashboard')
@login_required
def dash():
    return render_template("dashboard.html")
# This route provides an about me page for me the creator.


@app.route('/about')
@login_required
def about():
    return render_template("about.html")
# This route provides contact links. Not much going on here.


@app.route('/contact')
@login_required
def contact():
    return render_template("contact.html")

# This route provides shows all the currencies for the user if any.


@app.route('/currencies')
@login_required
def currencies():
    Currencies = UserCurrency.query.filter_by(id=current_user.id).all()
    print(Currencies)
    return render_template("currencies.html", Currencies=Currencies)

# This route is the main starter view of the app and contains info from the other sections


@app.route('/stocks')
@login_required
def stocks():
    # We want the price of 5+ stocks
    # http://finance.google.com/finance/info?client=ig&q=NASDAQ%3AAAPL,GOOG,MSFT,AMZN,TWTR
    if Stock.query.first() is None:
        print("No stock data found in DB")
        request = requests.get(
            'http://finance.google.com/finance/info?client=ig&q=NASDAQ%3AAAPL,GOOG,MSFT,AMZN,TWTR,EA,FB,NVDA,CSCO')
        # We need to change encoding as this API uses ISO and i use utf-8 everywhere else
        request.encoding = 'utf-8'
        # The response object contains some characters at start that we cant parse. Trim these off
        o = request.text[4:]
        # After we trim the characters, turn back into JSON
        result = json.loads(o)
        for i in result:
            # Now! Thats what I call Pythonic
            u = Stock(ticker=i['t'], last=i['l'], market=i['e'],
                      timestamp=datetime.datetime.now())
            db.session.add(u)

        db.session.commit()
    else:
        print("Found stock data in DB")
        # do something
    # query db for stocks
    Stocks = UserStocks.query.filter_by(id=current_user.id).all()

    # pass into html using render_template
    return render_template("stocks.html", Stocks=Stocks)


@app.route('/addNewStock', methods=['POST'])
def addNewStock():
    amount = request.form['Amount']  # Amount taken from posted form
    ticker = request.form['Ticker'].upper()  # Ticker taken from posted form
    queriedStock = Stock.query.filter_by(
        ticker=ticker).first()  # query the db for currency
    # Fiat is a term for financials i.e Euro, Dollar
    fiat = requests.get('http://api.fixer.io/latest?base=USD')
    usd2fiat = fiat.json()
    queriedCur = UserStocks.query.filter_by(
        ticker=ticker, id=current_user.id).first()

    if queriedStock is not None:
        if queriedCur is not None:
            queriedCur.amount += Decimal(amount)
            queriedCur.timestamp = datetime.datetime.now()
            print("Currency amount updated in DB")
        else:
            me = UserStocks(amount=float(amount), id=current_user.id, ticker=queriedStock.ticker, market=queriedStock.market, last=queriedStock.last, timestamp=datetime.datetime.now(), priceInUSD=((float(
                queriedStock.last)*float(amount))), priceInEUR=(((float(queriedStock.last)*float(amount))*float(usd2fiat['rates']['EUR']))), priceInCHY=(((float(queriedStock.last)*float(amount)) * float(usd2fiat['rates']['CNY']))))

            db.session.add(me)
            print("Currency added to DB")
        db.session.commit()
    else:
        flash('Unrecognised Ticker. Please select one of the supported tickers')
        print('Unrecognised Ticker. Please select one of the supported tickers')
    return redirect(url_for('stocks'))


# This route is used when a user adds a new currency. Info is submitted to server via POST.
# Removed Get method. Design Principle from John Healy. Use only what you need.


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
    
    
    
symbols = commons.short_list
windows = commons.windows
style = matplotlib.style.use('dark_background')
plt.style.use('dark_background')
# create our custom_dark theme from the plotly_dark template
plt_io.templates["custom_dark"] = plt_io.templates["plotly_dark"]

# set the paper_bgcolor and the plot_bgcolor to a new color
plt_io.templates["custom_dark"]['layout']['paper_bgcolor'] = '#30404D'
plt_io.templates["custom_dark"]['layout']['plot_bgcolor'] = '#30404D'

# you may also want to change gridline colors if you are modifying background
plt_io.templates['custom_dark']['layout']['yaxis']['gridcolor'] = '#4f687d'
plt_io.templates['custom_dark']['layout']['xaxis']['gridcolor'] = '#4f687d'

sns.set_style('whitegrid')


class CongestionIndexTrader:
    def __init__(self, symbol, start, now, interval, windows):
        self.symbol = symbol
        self.start = start
        self.now = now
        self.interval = interval
        self.windows = windows
        self.df = None

    def IMO_function(self, indicator, n):
        LL = indicator.rolling(window=n).min()
        HH = indicator.rolling(window=n).max()
        IMO_val = (indicator - LL) / (HH - LL) * 100
        return IMO_val.ewm(span=2, min_periods=3).mean()

    def signal_gen(self, IMO_df):
        Trade_ID = np.where(
            ((IMO_df < 75) & (IMO_df.shift(1) > 75)) | (IMO_df < 25), -1, 0)
        Trade_ID = np.where(
            ((IMO_df > 25) & (IMO_df.shift(1) < 25)) | (IMO_df > 75), 1, Trade_ID)
        return Trade_ID

    def calculate_window_indicators(self):
        for window in self.windows:
            HH = self.df['High'].rolling(window=window).max()
            LL = self.df['Low'].rolling(window=window).min()
            self.df[f'CI_{window}'] = self.df['Close'].pct_change(
                window) / (HH.pct_change(window) + 1e-9) * 100
            self.df[f'IDX_{window}'] = (
                self.df['Close'] - LL) / (HH - LL) * 100
            self.df[f'CI_IMO_{window}'] = self.IMO_function(
                self.df[f'CI_{window}'], window)
            self.df[f'IDX_IMO_{window}'] = self.IMO_function(
                self.df[f'IDX_{window}'], window)
            self.df[f'CI_signal_{window}'] = self.signal_gen(
                self.df[f'CI_IMO_{window}'])
            self.df[f'IDX_signal_{window}'] = self.signal_gen(
                self.df[f'IDX_IMO_{window}'])
            self.df[f'CI_trend_{window}'] = np.where(self.df[f'CI_{window}'] > 20, "Bullish",
                                                     np.where(self.df[f'CI_{window}'] < -20, "Bearish", "Congested"))

    def snapshot(self):
        self.df = yf.download(self.symbol, start=self.start,
                              end=self.now, interval=self.interval)
        self.df['date'] = self.df.index
        self.df['returns'] = np.log(
            self.df['Adj Close'] / self.df['Adj Close'].shift(1))
        self.df['log_returns'] = np.log(
            self.df['Close'] / self.df['Close'].shift(1))
        self.df['TR'] = abs(self.df['High'] - self.df['Low'])
        self.calculate_window_indicators()
        return self.df.fillna(0)

    # ... (backtest_and_analyze method remains unchanged) ...

    def plot_trends_and_signals(self, window):
        # Ensure the DataFrame is not empty
        if self.df is None or self.df.empty:
            print("DataFrame is empty. No data to plot.")
            return

        # Slice the DataFrame to the last 'slice_' rows
        slice_ = 255
        df_slice = self.df[-slice_:]

        # Calculate the buffer for y-axis limits
        buffer_percent = 0.05  # 5% buffer above and below the price range
        price_min = df_slice['Close'].min()
        price_max = df_slice['Close'].max()
        price_buffer = (price_max - price_min) * buffer_percent

        # Plot the closing price
        # Save the figure and axis reference
        fig, ax = plt.subplots(figsize=(14, 7))
        ax.plot(df_slice.index, df_slice['Close'],
                label='Closing Price', color='skyblue')

        # Plot CI_signal as markers
        ci_long_signals = df_slice[df_slice[f'CI_signal_{window}'] > 0]
        ci_short_signals = df_slice[df_slice[f'CI_signal_{window}'] < 0]
        ax.scatter(ci_long_signals.index,
                   ci_long_signals['Close'], label='CI Long Signal', marker='^', color='lime')
        ax.scatter(ci_short_signals.index,
                   ci_short_signals['Close'], label='CI Short Signal', marker='v', color='red')

        # Plot IDX_signal as markers
        idx_long_signals = df_slice[df_slice[f'IDX_signal_{window}'] > 0]
        idx_short_signals = df_slice[df_slice[f'IDX_signal_{window}'] < 0]
        ax.scatter(idx_long_signals.index,
                   idx_long_signals['Close'], label='IDX Long Signal', marker='o', color='aqua', alpha=0.5)
        ax.scatter(idx_short_signals.index,
                   idx_short_signals['Close'], label='IDX Short Signal', marker='x', color='fuchsia', alpha=0.5)

        # Highlight CI_trend areas with more opaque colors
        bullish_trend = df_slice[df_slice[f'CI_trend_{window}'] == 'Bullish']
        bearish_trend = df_slice[df_slice[f'CI_trend_{window}'] == 'Bearish']
        congested_trend = df_slice[df_slice[f'CI_trend_{window}'] == 'Congested']

        # Plot bullish trend areas
        ax.fill_between(
        bullish_trend.index, bullish_trend['Close'], color='green', alpha=0.3, label='Bullish Trend')
        # Plot bearish trend areas
        ax.fill_between(
        bearish_trend.index, bearish_trend['Close'], color='red', alpha=0.5, label='Bearish Trend')
        # Plot congested trend areas
        ax.fill_between(congested_trend.index,
                        congested_trend['Close'], color='orange', alpha=0.3, label='Congested Trend')

        # Customize and show the plot
        ax.set_title(
        f'Closing Price with CI and IDX Signals for {self.symbol}')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price')
        ax.legend()

        # Set y-axis limits with buffer
        ax.set_ylim(price_min - price_buffer, price_max + price_buffer)

        # Save the figure
        chart_filename = f'/workspaces/Congestion/charts/trading_chart_{
        self.symbol}.jpg'
        fig.savefig(chart_filename)
        plt.close(fig)  # Close the figure to free up memory


            # %%
"""Plot"""
# Set up date variables
now= datetime.datetime.now()
start_daily= now - datetime.timedelta(days=2500)
start_hourly= now - datetime.timedelta(days=720)
start_quarter= now - datetime.timedelta(days=80)

# Set the style to 'dark_background'
plt.style.use('dark_background')
# Usage
# Initialize a DataFrame to store the best window for each symbol
best_windows_df= pd.DataFrame(
columns=['Symbol', 'Best Window', 'Performance'])
windows= commons.windows

for symbol in short_list:
    # Initialize the trader with the necessary parameters
    trader= CongestionIndexTrader(
    symbol=symbol, start='2022-01-01', now=now, interval='1d', windows=windows)

    # Download the data and calculate indicators
    df_snapshot= trader.snapshot()
    df= pd.DataFrame(df_snapshot)

    # Plot trends and signals
    # Use the first window size for plotting
    trend_path= f"/workspaces/Congestion/db/csv/{symbol}.csv"
    df.to_csv(trend_path)
    # Assuming this method exists in your class
    trader.plot_trends_and_signals(window=20)
    