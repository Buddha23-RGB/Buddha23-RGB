#%%
from stockdashboard import routes
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_caching import Cache
from flask_marshmallow import Marshmallow
from flask_bcrypt import Bcrypt
from flask_login import LoginManager
from app import create_app
import os
from app import app
# from app.routes.routes import stocks_blueprint


# app.register_blueprint(stocks_blueprint, url_prefix='/stocks')

production = os.environ.get("PRODUCTION", False)
db = SQLAlchemy()
cache = Cache()
ma = Marshmallow()
bcrypt = Bcrypt()
login_manager = LoginManager()

def create_app():
    app = Flask(__name__)
    app.config.from_pyfile('config.cfg')

    db.init_app(app)
    cache.init_app(app)
    ma.init_app(app)
    bcrypt.init_app(app)
    login_manager.init_app(app)

    login_manager.login_view = 'login'
    login_manager.login_message_category = 'info'
    login_manager.login_message = "You must login to access this feature."
    login_manager.session_protection = "weak"

    with app.app_context():
        db.create_all()

    return app


app = create_app()

# %%
# Path: stockdashboard/routes.py
# from stockdashboard import routes  # Moved to the bottom

    
# %%
# !pip install waitress


@app.route('/')
def home():
    # Get the next image file path
    image_file = next(image_cycle)

    # Load symbols data
    symbols = load_symbols()  # Replace with your actual function to load symbols

    return render_template('index.html', image_file=image_file, symbols=symbols)


@app.route('/images/<filename>')
def send_image(filename):
    return send_from_directory(image_dir, filename)

if __name__ == '__main__':
    if production:
        app.run(debug=True)
    else:
        app.run(host='127.0.0.1', port=8001, debug=True)
    from waitress import serve
    serve(app, host='0.0.0.0', port=8080)

# %%
