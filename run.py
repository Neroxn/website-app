import os
import numpy as np  
import pandas as pd
from flask import Flask, request, redirect, url_for,render_template
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = 'C:\\Users\\kargi\\Flask Practi e\\basic-upload\\datasets'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif', 'log','csv'])

#Global variables that can bee accesed
df = pd.DataFrame()
selected = []
app = Flask(__name__) 
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

#Check if file is valid
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


#Upload file
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    global df
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            file_path = UPLOAD_FOLDER + "\\" + file.filename

            #Read file and go to next 
            df = pd.read_csv(file_path,encoding='utf-8', index_col=0)
            print(df.columns)
            return redirect(url_for('select_variables',
                                    filename=filename))
                                    
    return render_template("upload_file.html")

#Select x-variables among checkboxes
@app.route("/select_variables", methods = ["GET","POST"])
def select_variables():
    global selected
    if request.method == 'POST':
        selected = request.form.getlist('hello')
        return redirect(url_for('select_y'))

    return render_template("select_variables.html", df = df)

#Select y-variables among checkboxes
@app.route("/select_y",methods = ["GET","POST"])
def select_y():
    if request.method == 'POST':
        return """ SIKE """
    df2 = df.drop(selected, axis = 1)
    return render_template("select_y_variable.html", df = df2)
if __name__ == "__main__":
    app.run(debug=True)