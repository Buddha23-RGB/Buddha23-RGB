#%%
from app import create_app
import os
from app import app
# from app.routes.routes import stocks_blueprint


# app.register_blueprint(stocks_blueprint, url_prefix='/stocks')

production = os.environ.get("PRODUCTION", False)


    
# %%
# !pip install waitress
app = create_app()


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
