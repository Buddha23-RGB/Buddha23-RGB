#%%
!pip install Flask 
!pip install SocketIO
!pip install flask_socketio
!pip install flask-socketio
from flask_socketio import SocketIO
from flask_sqlalchemy import SQLAlchemy
from flask import Flask
import sqlite3
from sqlite3 import Error
import os
from dotenv import load_dotenv

load_dotenv('C:/Users/joech/OneDrive/Documents/Buddha23-RGB/FINAL_QI_2025/.gethub/.env')
db_file = r"C:\\Users\\joech\\OneDrive\\Documents\\Buddha23-RGB\\FINAL_QI_2025\\db\\stock.db"

app.config['SQLALCHEMY_DATABASE_URI'] = DB_URI
db = SQLAlchemy(app)


def connect_to_database():
    connection = sqlite3.connect(DB_FILE_PATH)
    cursor = connection.cursor()

    cursor.execute("SELECT 1")
    records = cursor.fetchall()
    print("Connected to database successfully")

    return connection, records


connection, records = connect_to_database()

cursor = connection.cursor()
cursor.execute("SELECT * FROM sqlite_master WHERE type='table';")
connection.close()
# specify your db file path


def create_connection(db_file):
    """ create a database connection to a SQLite database """
    conn = None;
    try:
        conn = sqlite3.connect(db_file)
        print(f'Successful connection with {db_file}')
    except Error as e:
        print(e)
    return conn

def create_table(conn, create_table_sql):
    """ Create a table from the create_table_sql statement """
    try:
        c = conn.cursor()
        c.execute(create_table_sql)
    except Error as e:
        print(e)

# create a database connection
conn = create_connection(db_file)

# create users table sql command
create_users_table_sql = """CREATE TABLE IF NOT EXISTS users (
                                        id integer PRIMARY KEY,
                                        username text NOT NULL,
                                        password text NOT NULL,
                                        cash_holdings real
                                    ); """
# create a users table
create_table(conn, create_users_table_sql)

# create portfolio table sql command
create_portfolio_table_sql = """CREATE TABLE IF NOT EXISTS portfolio (
                                        id integer PRIMARY KEY,
                                        user_id integer,
                                        stock_symbol text NOT NULL,
                                        quantity integer,
                                        FOREIGN KEY (user_id) REFERENCES users (id)
                                    ); """
# create a portfolio table
create_table(conn, create_portfolio_table_sql)

# close the connection
conn.close()
# Config classes


basedir = os.path.abspath(os.path.dirname(__file__))

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY') or os.urandom(24)
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL') or \
    'sqlite:///' + os.path.join(basedir, 'stock.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
socketio = SocketIO(app)


class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), index=True, unique=True)
    password = db.Column(db.String(128))
    cash_holdings = db.Column(db.Float)


class Portfolio(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    stock_symbol = db.Column(db.String(10))
    quantity = db.Column(db.Integer)

    user = db.relationship(
        'User', backref=db.backref('portfolio', lazy='dynamic'))


@socketio.on('update_portfolio')
def handle_update_portfolio(message):
    # Update portfolio based on real-time data
    # This is just a placeholder. You need to implement the logic to update the portfolio based on real-time data.
    pass


if __name__ == '__main__':
    db.create_all()
    socketio.run(app)
