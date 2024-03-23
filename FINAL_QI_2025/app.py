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