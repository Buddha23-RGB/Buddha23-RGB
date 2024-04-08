#!/bin/bash
export FLASK_APP=app.py
export FLASK_ENV=development
docker build -t FINAL_QI_2025 .
docker run -p 3000:3000 -d FINAL_QI_2025
python -m ensurepip --upgrade
pip install virtualenv
python -m venv env
pipenv install requests
pipenv install flask_caching
pipenv install flask_marshmallow
pipenv install flask_marshmallow
pipenv install flask_caching
pipenv install Flask-Caching as flask_caching
pipenv install flask_bcrypt
pipenv install flask-bcrypt
pipenv install -r requirements.txt
pipenv update
pipenv shell
flask run


git checkout Buddha23-RGB-QI
# Make your changes
git add .
git commit -m "Your commit message Aplir8"
git push origin Buddha23-RGB-QI

env/Scripts/Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
set FLASK_APP "app.py"
set FLASK_DEBUG=1
flask run app.py
uvicorn app:app --reload