#!/bin/bash

# Set environment variables for Flask
export FLASK_APP=app.py
export FLASK_ENV=development

# Build and run Docker container
docker build -t congestion_devcontainer-db-1 .
docker run -p 3000:3000 -d congestion_devcontainer-db-1

# Display running Docker containers
docker ps

# Ensure pip is upgraded
python -m ensurepip --upgrade

# Install virtualenv and create a new virtual environment
pip install virtualenv
python -m venv env

# Activate the virtual environment
# source env/bin/activate
env/Scripts/Activate.ps1
# Install Python packages using pip instead of pipenv
pip install requests flask_caching flask_marshmallow flask_bcrypt

pipenv install -r requirements.txt
pip install --upgrade pipenv
pipenv shell
pipenv run

# Git commands
git checkout Buddha23-RGB-QI
git add .
git commit -m "april12"

# Run Flask application
flask run# Run Flask application

#!/bin/bash
export FLASK_APP=app.py
export FLASK_ENV=development
docker build -t congestion_devcontainer-db-1 .
docker run -p 3000:3000 -d congestion_devcontainer-db-1
docker ps
python -m ensurepip --upgrade
pip install virtualenv
python -m venv env
git checkout Buddha23-RGB-QI
# Make your changes
git add .
git commit -m "Your commit message Aplir8"
flask run


# git push origin Buddha23-RGB-QI

env/Scripts/Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
set FLASK_APP "app.py"
set FLASK_DEBUG=1
flask run app.py
uvicorn app:app --reload