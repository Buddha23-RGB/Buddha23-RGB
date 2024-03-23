#!/bin/bash
export FLASK_APP=app.py
export FLASK_ENV=development
docker build -t FINAL_QI_2025 .
docker run -p 3000:3000 -d FINAL_QI_2025
python -m ensurepip --upgrade
pip install virtualenv
python -m venv env

env/Scripts/Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
set FLASK_APP "app.py"
set FLASK_DEBUG=1
flask run app.py
uvicorn app:app --reload