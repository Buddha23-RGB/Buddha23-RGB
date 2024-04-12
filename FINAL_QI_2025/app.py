#%%
# !pip install flask-login
# !pip install Flask-Login
# !pip install Flask-Login
# #%%
# %pip install flask-login
# Import standard libraries
from flask_login import login_required
from flask_login import current_user
from flask import Flask, render_template, url_for, flash, redirect, request
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, login_user, logout_user, UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from forms import RegistrationForm
from itertools import cycle
import os
import json
from dotenv import load_dotenv
import pandas as pd
import commons
from commons import *
# Load environment variables from .env file if it exists
load_dotenv(
    "C:\\Users\\joech\\OneDrive\\Documents\\Buddha23-RGB\\FINAL_QI_2025\\.github\\.env")

# Directory containing images
image_dir = "C:/Users/joech/OneDrive/Documents/Buddha23-RGB/FINAL_QI_2025/db/charts"

# Get all image files in the directory
image_files = [f for f in os.listdir(image_dir) if f.endswith(
    ('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]

# Cycle through the images (loop back to the start when reaching the end)
image_cycle = cycle(image_files)

# Load config
with open('config.json') as f:
    config = json.load(f)

# Set up global variables
root_dir = config['root_dir']

# Initialize the SQLAlchemy instance with no app
db = SQLAlchemy()


class User(UserMixin, db.Model):

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)

    def __repr__(self):
        return '<User %r>' % self.username
def create_app():
    app = Flask(__name__)
    app.config['SECRET_KEY'] = os.getenv(
        'SECRET_KEY', 'your_default_secret_key')
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///C:\\Users\\joech\\OneDrive\\Documents\\Buddha23-RGB\\FINAL_QI_2025\\db\\stock.db'

    # Then use init_app to set the app for the SQLAlchemy instance
    db.init_app(app)

    with app.app_context():
        db.create_all()

    login_manager = LoginManager()
    login_manager.init_app(app)

    return app, db


app, db = create_app()

login_manager = LoginManager(app)
login_manager.login_view = 'login'


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


@app.route("/register", methods=['GET', 'POST'])
def register():
    form = RegistrationForm()
    if form.validate_on_submit():
        hashed_password = generate_password_hash(
            form.password.data, method='sha256')
        user = User(username=form.username.data,
                    email=form.email.data, password=hashed_password)
        db.session.add(user)
        db.session.commit()
        flash('Your account has been created! You are now able to log in', 'success')
        return redirect(url_for('login'))
    return render_template('register.html', title='Register', form=form)


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        user = User.query.filter_by(username=username).first()

        if user is None or not check_password_hash(user.password, password):
            return redirect(url_for('login'))

        login_user(user)

        return redirect(url_for('index'))

    return render_template('login.html')


@app.route('/logout')
def logout():
    logout_user()
    return redirect(url_for('index'))


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


@app.route('/table_css')
def table_css():
    return render_template('tables/table_css.html')

@app.route('/dashboard')
@login_required
def dashboard():
    return render_template('dashboard.html', name=current_user.username)
@app.route('/')
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
# @login_manager.user_loader
# def load_user(user_id):
#     return User.get(user_id)
# # @app.route('/', methods=['GET', 'POST'])
# def portfolio_dashboard():
#     symbols = [{'name': 'AAPL'}, {'name': 'GOOG'}, {
#         'name': 'MSFT'}]  # Replace with your actual data
#     selected_symbol = None

#     if request.method == 'POST':
#         selected_symbol = request.form.get('symbol')

#     return render_template('stock_breakdown.html', symbols=symbols, selected_symbol=selected_symbol)

# @app.route('/')
# def home():
#     # Get the next image file path
#     image_file = next(image_cycle)

#     # Load symbols data
#     symbols = load_symbols()  # Replace with your actual function to load symbols

#     return render_template('index.html', image_file=image_file, symbols=symbols)


# @app.route('/images/<filename>')
# def send_image(filename):
#     return send_from_directory(image_dir, filename)
