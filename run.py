import os
import numpy as np  
import pandas as pd
from bokeh.models import ColumnDataSource, Div, Select, Slider, TextInput
from bokeh.io import curdoc
from bokeh.resources import INLINE
from bokeh.embed import components
from bokeh.plotting import figure, output_file, show
from flask import Flask, request, redirect, url_for,render_template
from werkzeug.utils import secure_filename
from utils import *
from modelTrain import *
from preprocess import *
from sklearn.model_selection import train_test_split
UPLOAD_FOLDER = 'C:\\Users\\kargi\\Flask Practi e\\website-app-main-2\\datasets'
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
                           link_column="Patient ID", zip=zip, isLoaded = isLoaded, rowS = df.shape[0], colS = df.shape[1])
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

    if(len(df) != 0):
        dtypes, cols = groupColumns(df)
    else:
        dtypes = []
        cols = []
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
    
    if(len(df) != 0):
        dtypes, cols = groupColumns(possibleDf)
    else:
        dtypes = []
        cols = []
    return render_template("select_variables.html",types = dtypes, columns = cols)


@app.route("/selectAlgo", methods = ["GET","POST"])
def selectAlgo():
    if request.method == 'POST':
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
        #Check whether kernel is valid
        if not request.form['kernel'] in ['linear', 'rbf', 'poly','sigmoid']:
            flash('Invalid kernel, kernel must be one of them: linear, rbf, poly, sigmoid')
            return render_template("algorithms/SVM.html")
        else:
            kernel = request.form['kernel']
            
        #Check whether C is valid
        try:
            if float(request.form['C']) <= 0 :
                flash('C must be positive real number')
                return render_template("algorithms/SVM.html")
            else:
                C = request.form['C']
        except:
            flash('C must be a float number')
            return render_template("algorithms/SVM.html")
            
        #Check whether gamma is valid
        if(kernel == 'linear'):
            gamma = 'scale'
        try:
            if request.form['gamma'] == 'scale':
                gamma = 'scale'
            elif request.form['gamma'] == 'auto':
                gamma = 'auto'
            elif float(request.form['gamma']) == 0:
                flash('Gamma must be different from 0')
                return render_template("algorithms/SVM.html")
            else:
                gamma = float(request.form['gamma'])
        except:
            flash('Gamma must be auto, scale, or a float number')
            return render_template("algorithms/VM.html")
        
        #Check whether degree is valid
        if(kernel != 'poly' and request.form['degree'] != ''):
            flash('Poly must be defined if kernel is polynomial')
            return render_template("algorithms/SVM.html")
        if(kernel == 'poly'):
            try:
                if int(request.form['degree']) <= 0:
                    flash('Degree must be positive integer')
                    return render_template("algorithms/SVM.html")
                else:
                    degree = int(request.form['degree'])
                
            except:
                flash('Degree must be a positive integer')
                return render_template("algorithms/SVM.html")
        else:
            degree = 3
            
        #Parameters are valid, train time
        df2 = df[selectedX+selectedY]
        df2 = dropNanAndDuplicates(df2, 0.75)
        df2, encoderArr = stringEncoder(df2, df2.loc[:, df2.dtypes == object].columns)
        df2, scaler, scalerY = scale(df2, selectedY)
        trainX, testX, trainY, testY = train_test_split(df2[selectedX], df2[selectedY], 
                                                        test_size= 0.15, shuffle= True)
        
        model = applySVM(trainX, trainY, kernel= kernel, c= float(C), gamma= gamma, degree= float(degree))
        
        #Train is done, predict time
        result = model.predict(testX).reshape((len(testX),-1))
        print(result.shape)
        result = scalerY.inverse_transform(result)
        if encoderArr:
            result = stringDecoder(result, encoderArr, selectedY)
        
        return redirect(url_for('results', actual= testY, prediction= result))
        
    return render_template("algorithms/SVM.html")

@app.route("/RandomForest", methods = ["GET","POST"])
def RandomForest():
    if request.method == 'POST':
    
        #Check whether number of estimator is valid
        try:
            if int(request.form['numberEstimator']) <= 0 :
                flash('Number of estimator must be a positive integer')
                return render_template("algorithms/RandomForest.html")
            else:
                numberEstimator = int(request.form['numberEstimator'])
        except:
            flash('Number of estimator must be an integer')
            return render_template("algorithms/RandomForest.html")
            
        #Check whether maxDepth is valid
        try:
            if request.form['maxDepth'] == 'None':
                maxDepth = 'None'
            elif int(request.form['maxDepth']) <= 0 :
                flash('Max depth must be a positive integer')
                return render_template("algorithms/RandomForest.html")
            else:
                maxDepth = int(request.form['maxDepth'])
        except:
            flash('Max depth must be an integer or None')
            return render_template("algorithms/RandomForest.html")
            
        #Check whether minimum samples leaf is valid
        try:
            if int(request.form['minSamplesLeaf']) <= 0 :
                flash('Minimum samples leaf must be a positive integer or float')
                return render_template("algorithms/RandomForest.html")
            else:
                minSamplesLeaf = int(request.form['minSamplesLeaf'])
        except:
            try:
                if float(request.form['minSamplesLeaf']) <= 0 :
                    flash('Minimum samples leaf must be a positive integer or float')
                    return render_template("algorithms/RandomForest.html")
                else:
                    minSamplesLeaf = float(request.form['minSamplesLeaf'])
            except:
                flash('Minimum samples leaf must be an integer or float')
                return render_template("algorithms/RandomForest.html")
                
        #Parameters are valid, train time
        df2 = df[selectedX+selectedY]
        df2 = dropNanAndDuplicates(df2, 0.75)
        df2, encoderArr = stringEncoder(df2, df2.loc[:, df2.dtypes == object].columns)
        trainX, testX, trainY, testY = train_test_split(df2[selectedX], df2[selectedY], 
                                                        test_size= 0.15, shuffle= True)
        model = applyRandomForest(trainX, trainY, numberEstimator= numberEstimator, maxDepth = maxDepth, minSamplesLeaf=minSamplesLeaf)
        
        #Train is done, predict time
        result = model.predict(testX).reshape((len(testX),-1))
        if encoderArr:
            result = stringDecoder(result, encoderArr, selectedY)
        
        return redirect(url_for('results', actual= testY, prediction= result))
    return render_template("algorithms/RandomForest.html")

@app.route("/Adaboost", methods = ["GET","POST"])
def Adaboost():
    if request.method == 'POST':
    
        #Check whether number of estimator is valid
        try:
            if int(request.form['numberEstimator']) <= 0 :
                flash('Number of estimator must be a positive integer')
                return render_template("algorithms/Adaboost.html")
            else:
                numberEstimator = int(request.form['numberEstimator'])
        except:
            flash('Number of estimator must be an integer')
            return render_template("algorithms/Adaboost.html")
            
        #Check whether learning rate is valid
        try:
            if float(request.form['learningRate']) <= 0 :
                flash('Learning rate must be a positive float')
                return render_template("algorithms/RandomForest.html")
            else:
                learningRate = float(request.form['learningRate'])
        except:
            flash('Number of estimator must be an float')
            return render_template("algorithms/RandomForest.html")
            
        #Check whether loss is valid
        if not request.form['loss'] in ['linear', 'square', 'exponential']:
            flash('Invalid loss, loss must be one of them: linear, square, exponential')
            return render_template("algorithms/RandomForest.html")
        else:
            loss = request.form['loss']
            
        #Parameters are valid, train time
        df2 = df[selectedX+selectedY]
        df2 = dropNanAndDuplicates(df2, 0.75)
        df2, encoderArr = stringEncoder(df2, df2.loc[:, df2.dtypes == object].columns)
        trainX, testX, trainY, testY = train_test_split(df2[selectedX], df2[selectedY], 
                                                        test_size= 0.15, shuffle= True)
        model = applyAdaBoost(trainX, trainY, numberEstimator= numberEstimator, learningRate = learningRate, loss=loss)
        
        #Train is done, predict time
        result = model.predict(testX).reshape((len(testX),-1))
        if encoderArr:
            result = stringDecoder(result, encoderArr, selectedY)
        
        return redirect(url_for('results', actual= testY, prediction= result))
            
    return render_template("algorithms/Adaboost.html")

@app.route("/results", methods = ["GET","POST"])
def results():
    if request.method == "POST":
        return "To be continued"

    return "To be continued"
        
    prediction = request.args['prediction']
    actual = request.args['actual']
    return "To be continued"
        
    return render_template("algorithms/Adaboost.html")
    
@app.route('/visualize',methods = ["GET","POST"])
def visualize():
    print(request.method)
    if request.method == 'POST':

        if request.form['selector'] == "scatter":
            return redirect(url_for('scatter_graph'))

        elif request.form['selector'] == "corr":
            return redirect(url_for('correlation_graph'))

        elif request.form['selector'] == "pie":
            return redirect(url_for('pie_graph'))

        elif request.form['selector'] == "dist":
            return redirect(url_for('dist_graph'))

    return render_template('graphs/visualize.html', graphSelected = False)
    
@app.route('/scatter_graph', methods = ["GET","POST"])
def scatter_graph():
    global df
    if request.method == "POST":
        if 'parameters' in request.form:
            selected_features = request.form.getlist('parameters')
            
        else:
            return render_template('graphs/scatter_plot.html',columns = df.columns,error = "Please select parameters to process.")

        if len(selected_features) >= 10:
            return render_template('graphs/scatter_plot.html',columns = df.columns,error = "You have selected more than 10 features!. Please select again.")

        elif len(selected_features) <= 2:
            return render_template('graphs/scatter_plot.html',columns = df.columns,error = "You have selected less than 2 features!. Please select again.")
        
        else:
            return scatter_matrix(df,selected_features)
    return render_template('graphs/scatter_plot.html',columns = df.columns)


@app.route('/correlation_graph', methods = ["GET","POST"])
def correlation_graph():
    global df
    if request.method == "POST":
        if 'parameters' in request.form:
            selected_features = request.form.getlist('parameters')
            return correlation_plot(df.select_dtypes(exclude = ['object']),selected_features)
            
        else:
            return render_template('graphs/correlation_plot.html',columns = df.select_dtypes(exclude = ['object']).columns,error = "Please select parameters to process.")

    return render_template('graphs/correlation_plot.html',columns = df.select_dtypes(exclude = ['object']).columns)


@app.route('/pie_graph', methods = ["GET","POST"])
def pie_graph():
    global df
    if request.method == "POST":
        print(request.form)
        if "selected_parameter" in request.form:
            return pie_plot(df.select_dtypes(include = ["object"]),request.form['selected_parameter'])
        else:
            return render_template('graphs/pie_plot.html',columns =  df.select_dtypes(include = ["object"]).columns,error = "Please select parameters to process.")

    return render_template('graphs/pie_plot.html',columns = df.select_dtypes(include = ["object"]).columns)

@app.route('/dist_graph', methods = ["GET","POST"])
def dist_graph():
    global df
    if request.method == "POST":
        print(request.form)
        if 'selected_parameter' in request.form:
            numberBin = 20 if 'numberBin' not in request.form else int(request.form['numberBin'])
            return dist_plot(df.select_dtypes(exclude = ['object']),request.form['selected_parameter'],numberBin)
        else:
            return render_template('graphs/dist_plot.html',columns =df.select_dtypes(exclude = ['object']).columns, error = "Please choose the parameter for histogram!")
    return render_template('graphs/dist_plot.html',columns = df.select_dtypes(exclude = ['object']).columns)

        
if __name__ == "__main__":
    app.run(debug=True)
