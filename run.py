import os
import numpy as np  
import pandas as pd
from flask import Flask, request, redirect, url_for,render_template
from werkzeug.utils import secure_filename
from helper.utils import *

UPLOAD_FOLDER = 'C:\\Users\\kargi\\Flask Practi e\\basic-upload\\datasets'
ALLOWED_EXTENSIONS = set(['txt', 'csv'])

#Global variables that can bee accesed
df = pd.DataFrame()
dataColumns = pd.DataFrame()
selectedX = []
selectedY = []
app = Flask(__name__) 
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    global df
    if request.method == 'POST':

        # check if the post request has the file part
        print(request.files)
        if 'file' not in request.files:
            #flash('No file part')
            return render_template("upload_file.html",errors = ['No file is submitted!'])
        file = request.files['file']

        # check if user does not send any file
        if file.filename == '':
            #flash('No selected file')
            return render_template("upload_file.html",errors = ['No file is selected!'])

        # a valid file is submitted. 
        if file and allowed_file(file.filename):
            # construct the file path
            filename = secure_filename(file.filename)
            file_path = UPLOAD_FOLDER + "\\" + file.filename
            print(file_path)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

            # get delimitter and qualifier form the form.
            delimitter = request.form['delimitter']
            qualifier = request.form['qualifier']

            if request.form.get('is-value-type'):
                assumption = True
            else:
                assumption = False

            # read the file
            dataTypes, dataColumns, df = load_dataset(file_path,delimitter=delimitter,qualifier = qualifier, assumption=assumption)

            #return redirect(url_for('select_variables',filename=filename))
            isLoaded = True
            return render_template("upload_file.html", column_names=df.columns.values, row_data=list(df.head(5).values.tolist()),
                           link_column="Patient ID", zip=zip, isLoaded = isLoaded)
        else:
            return render_template("upload_file.html",errors = ['Extension is not correct!'])
    else:                        
        return render_template("upload_file.html")

#Select x-variables among checkboxes
@app.route("/select_variables", methods = ["GET","POST"])
def select_variables():
    global selectedX,df
    if request.method == 'POST':
        selectedX = request.form.getlist('hello')
        return redirect(url_for('select_y'))

    dtypes, cols = groupColumns(df)
    return render_template("select_variables.html",types = dtypes, columns = cols)

#Select y-variables among checkboxes
@app.route("/select_y",methods = ["GET","POST"])
def select_y():
    global selectedY
    if request.method == 'POST':
        selectedY = request.form.getlist('hello')
        return redirect(url_for('selectAlgo'))

    finalColumnNamesY= []
    possibleDf =df.drop(selectedX,axis=1)
    
    dtypes, cols = groupColumns(possibleDf)
    return render_template("select_y_variable.html", types = dtypes, columns = cols)


@app.route("/visualize", methods = ["GET","POST"])
def visualize():
    return render_template("visualize.html")

@app.route("/selectAlgo", methods = ["GET","POST"])
def selectAlgo():
    if request.method == 'POST'
        selectedAlgo = request.form['selector']
        
        if selectedAlgo == 'SVM':
            return redirect(url_for('SVM'))
            
        elif selectedAlgo == 'RandomForest':
            return redirect(url_for('RandomForest'))
            
        elif selectedAlgo == 'Adaboost':
            return redirect(url_for('Adaboost'))
        
    return render_template("select_algo.html")

@app.route("/SVM", methods = ["GET","POST"])
def SVM():
    if request.method == 'POST':
        return "To be continued"
        
    return render_template("SVM.html")

@app.route("/RandomForest", methods = ["GET","POST"])
def RandomForest():
    if request.method == 'POST':
        return "To be continued"
        
    return render_template("RandomForest.html")

@app.route("/Adaboost", methods = ["GET","POST"])
def Adaboost():
    if request.method == 'POST':
        return "To be continued"
        
    return render_template("Adaboost.html")
    
if __name__ == "__main__":
    app.run(debug=True)
