# %%
import commons
from commons import *
import os
with open('config.json') as f:
    config = json.load(f)

root_dir = config['root_dir']
rel_path = config['rel_path']
table_path = config['table_path']
benchmarks = config['benchmarks']
short_list = config['short_list']
windows = config['windows']
# %%


# Define the base directory of the application
basedir = os.path.abspath(os.path.dirname(__file__))

# Config classes


class Config:
    # Generate a random secret key
    SECRET_KEY = os.environ.get('SECRET_KEY') or os.urandom(24)
    # Define the database URI for SQLAlchemy
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or \
        'sqlite:///' + os.path.join(basedir, 'stock.db')
    # Disable signal handling (which can consume memory)
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    # Email configuration for Flask-Mail (if needed)
    MAIL_SERVER = 'smtp.googlemail.com'
    MAIL_PORT = 587
    MAIL_USE_TLS = True
    MAIL_USERNAME = os.environ.get('EMAIL_USER')
    MAIL_PASSWORD = os.environ.get('EMAIL_PASS')


class DevelopmentConfig(Config):
    DEBUG = True
    # Set the full path for the development SQLite database
    SQLALCHEMY_DATABASE_URI = "C:\\Users\\joech\\OneDrive\\Documents\\Buddha23-RGB\\FINAL_QI_2025\\db\\stock.db"
