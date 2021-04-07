import os
import numpy as np  
import pandas as pd
from flask import Flask, request, redirect, url_for,render_template
from werkzeug.utils import secure_filename
from utils import *

UPLOAD_FOLDER = 'C:\\Users\\kargi\\Flask Practi e\\basic-upload\\datasets'
ALLOWED_EXTENSIONS = set(['txt', 'csv'])

#Global variables that can bee accesed
df = pd.DataFrame()
selected = []
app = Flask(__name__) 
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    global df
    
    if request.method == 'POST':

        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']

        # check if user does not send any file
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)

        # a valid file is submitted. 
        if file and allowed_file(file.filename):
            # construct the file path
            filename = secure_filename(file.filename)
            file_path = UPLOAD_FOLDER + "\\" + file.filename
            print(file_path)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

            delimitter = request.form['delimitter']
            qualifier = request.form['qualifier']
            if request.form.get('is-value-type'):
                assumption = True
            else:
                assumption = False
            print(assumption)
            # read the file
            dataTypes, dataColumns, df = load_dataset(file_path,delimitter=delimitter,qualifier = qualifier, assumption=assumption)
            return redirect(url_for('select_variables',filename=filename))
    else:                        
        return render_template("upload_file.html")

#Select x-variables among checkboxes
@app.route("/select_variables", methods = ["GET","POST"])
def select_variables():
    global selected
    if request.method == 'POST':
        selected = request.form.getlist('hello')
        return redirect(url_for('select_y'))

    return render_template("select_variables.html", df = df, columns = df.columns.sort_values())

#Select y-variables among checkboxes
@app.route("/select_y",methods = ["GET","POST"])
def select_y():
    if request.method == 'POST':
        return """ SIKE """
    df2 = df.drop(selected, axis = 1)
    return render_template("select_y_variable.html", df = df2, columns = df2.columns.sort_values())

if __name__ == "__main__":
    app.run(debug=True)
