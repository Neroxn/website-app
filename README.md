# Website Application for creating practical Machine Learning models.qx

## About website

## Requirements
- Required libraries are presented in the file `requirements.txt`. This important file can be used to create new virtual environment for Python - or for deploying the app.
- Used python version is : `Python 3.7.3`
## How to run?
- This website can be run using flask commands. 
  - Open any command prompt of your choice and set environment variables. Read the paragraph below for learning about environment variables.
  - Set the `secret_key` variable in the `app.py` carefully before deploying the app. In development process, this is not required.
  - Use the command `flask init-db` to initialize empty database and table for the user

Important environment variables that should be set before:
- `FLASK_APP` -- This is important for determining where is the application. For more information, <a href=https://flask.palletsprojects.com/en/2.0.x/cli/> see. </a>
- `FLASK_ENV` -- This variable is used for starting the application in `debug` mode .For more information, <a href=https://flask.palletsprojects.com/en/2.0.x/config/> see. </a>

## Resources
- Read <a href = https://flask.palletsprojects.com/en/2.0.x/> flask documentation </a> for more information about setting the environment variables, debugging and functions.
