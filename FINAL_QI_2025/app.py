#%%
!pip install flask-login
import logging
import pandas as pd
from flask import Flask, render_template
from configs import DevelopmentConfig
import configs
import mysql.connector
import commons
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
import sqlite3
import json
import sys

print(sys.executable)

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
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user

login_manager = LoginManager()
login_manager.init_app(app)
app = Flask(__name__)
@login_manager.user_loader
def load_user(user_id):
    return User.get(user_id)
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    # other fields...

    def get_id(self):
        return str(self.id)
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username).first()
        if user and user.check_password(password):
            login_user(user)
            return redirect(url_for('index'))
        else:
            flash('Invalid username or password')
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username).first()
@app.route('/')
def home():
    aapl_df = pd.read_csv(
        'C:/Users/joech/OneDrive/Documents/Buddha23-RGB/FINAL_QI_2025/db/tables/AAPL.csv')
    aapl_divergence_df = pd.read_csv(
        'C:/Users/joech/OneDrive/Documents/Buddha23-RGB/FINAL_QI_2025/db/tables/AAPL_divergence.csv')

    return render_template('index.html', tables=[aapl_df.to_html(classes='data', header="true"), aapl_divergence_df.to_html(classes='data', header='true')])


logging.basicConfig(filename='error.log', level=logging.DEBUG)
@app.route('/secret')
@login_required
def secret():
    return 'Only authenticated users are allowed!'
# ...

if __name__ == '__main__':
    try:
        app.run()
    except Exception as e:
        app.logger.error(f'Failed to start the app due to {e}')
#%%

aapl_df = pd.read_csv(
    'C:/Users/joech/OneDrive/Documents/Buddha23-RGB/FINAL_QI_2025/db/tables/AAPL.csv')

# %%
aapl_df.tail(20)
# %%
 aapl_divergence_df = pd.read_csv(
        'C:/Users/joech/OneDrive/Documents/Buddha23-RGB/FINAL_QI_2025/db/tables/AAPL_divergence.csv')
aapl_divergence_df.tail(20)

# %%
