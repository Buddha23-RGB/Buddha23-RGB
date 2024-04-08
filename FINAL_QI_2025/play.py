#%%
from flask import Flask, render_template

app = Flask(
    __name__, template_folder='C:/Users/joech/OneDrive/Documents/Buddha23-RGB/FINAL_QI_2025/templates')


@app.route('/')
def home():
    return render_template('table_css.html')


if __name__ == '__main__':
    app.run(debug=True)
#%%