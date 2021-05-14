import os
import numpy as np  
import pandas as pd
from bokeh.models import ColumnDataSource, Div, Select, Slider, TextInput
from bokeh.io import curdoc
from bokeh.resources import INLINE
from bokeh.embed import components
from bokeh.plotting import figure, output_file, show
from flask import Flask, request, redirect, url_for,render_template,Response
from werkzeug.utils import secure_filename
from utils import *
from modelTrain import *
from preprocess import *
from db import *
from workspace import *
from sklearn.model_selection import train_test_split

### These variables will be fixed later on as they are global and will cause errors. ###
df = pd.read_csv("DataCasiaEhr.csv")
dataColumns = pd.DataFrame()
selectedX = []
selectedY = []
user_log = []
model = None
graph = None
selectedModel = None
########################################################################################


def create_app(test_config = None):
    UPLOAD_FOLDER = 'C:\\Users\\kargi\\Flask Practi e\\website-app-main-2\\datasets'
    ALLOWED_EXTENSIONS = set(['txt', 'csv'])
    app = Flask(__name__) 
    DATABASE=os.path.join(app.instance_path, 'flaskr.sqlite')

    app.config.from_mapping(
        SECRET_KEY='dev',
        DATABASE=os.path.join(app.instance_path, 'flaskr.sqlite'),
        UPLOAD_FOLDER = UPLOAD_FOLDER
    )
    
    if test_config is None:
        # load the instance config, if it exists, when not testing
        app.config.from_pyfile('config.py', silent=True)
    else:
        # load the test config if passed in
        app.config.from_mapping(test_config)

    # ensure the instance folder exists
    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass


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
                #Check whether kernel is valid
                if not request.form['kernel'] in ['linear', 'rbf', 'poly','sigmoid']:
                    flash('Invalid kernel, kernel must be one of them: linear, rbf, poly, sigmoid')
                    return redirect(url_for('selectAlgo'))
                else:
                    kernel = request.form['kernel']
                
                #Check whether C is valid
                try:
                    if float(request.form['C']) <= 0 :
                        flash('C must be positive real number')
                        return redirect(url_for('selectAlgo'))
                    else:
                        C = request.form['C']
                except:
                    flash('C must be a float number')
                    return redirect(url_for('selectAlgo'))
                
                #Check whether gamma is valid
                if(kernel == 'linear'):
                    gamma = 'scale'
                else:
                    try:
                        if request.form['gamma'] == 'scale':
                            gamma = 'scale'
                        elif request.form['gamma'] == 'auto':
                            gamma = 'auto'
                        elif float(request.form['gamma']) == 0:
                            flash('Gamma must be different from 0')
                            return redirect(url_for('selectAlgo'))
                        else:
                            gamma = float(request.form['gamma'])
                    except:
                        flash('Gamma must be auto, scale, or a float number')
                        return redirect(url_for('selectAlgo'))
            
                #Check whether degree is valid
                if(kernel != 'poly' and request.form['degree'] != ''):
                    flash('Poly must be defined if kernel is polynomial')
                    return render_template("select_algo.html")
                if(kernel == 'poly'):
                    try:
                        if int(request.form['degree']) <= 0:
                            flash('Degree must be positive integer')
                            return redirect(url_for('selectAlgo'))
                        else:
                            degree = int(request.form['degree'])
                    
                    except:
                        flash('Degree must be a positive integer')
                        return redirect(url_for('selectAlgo'))
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
                result = scalerY.inverse_transform(result)
                if encoderArr:
                    result = stringDecoder(result, encoderArr, selectedY)
            
                return redirect(url_for('results', actual= testY, prediction= result))
                
            elif selectedAlgo == 'RandomForest':
                #Check whether number of estimator is valid
                try:
                    if int(request.form['numberEstimator']) <= 0 :
                        flash('Number of estimator must be a positive integer')
                        return redirect(url_for('selectAlgo'))
                    else:
                        numberEstimator = int(request.form['numberEstimator'])
                except:
                    flash('Number of estimator must be an integer')
                    return redirect(url_for('selectAlgo'))
                    
                #Check whether maxDepth is valid
                try:
                    if request.form['maxDepth'] == 'None':
                        maxDepth = 'None'
                    elif int(request.form['maxDepth']) <= 0 :
                        flash('Max depth must be a positive integer')
                        return redirect(url_for('selectAlgo'))
                    else:
                        maxDepth = int(request.form['maxDepth'])
                except:
                    flash('Max depth must be an integer or None')
                    return redirect(url_for('selectAlgo'))
                
                #Check whether minimum samples leaf is valid
                try:
                    if int(request.form['minSamplesLeaf']) <= 0 :
                        flash('Minimum samples leaf must be a positive integer or float')
                        return redirect(url_for('selectAlgo'))
                    else:
                        minSamplesLeaf = int(request.form['minSamplesLeaf'])
                except:
                    try:
                        if float(request.form['minSamplesLeaf']) <= 0 :
                            flash('Minimum samples leaf must be a positive integer or float')
                            return redirect(url_for('selectAlgo'))
                        else:
                            minSamplesLeaf = float(request.form['minSamplesLeaf'])
                    except:
                        flash('Minimum samples leaf must be an integer or float')
                        return redirect(url_for('selectAlgo'))
                            
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
                
            elif selectedAlgo == 'Adaboost':
                #Check whether number of estimator is valid
                try:
                    if int(request.form['numberEstimator']) <= 0 :
                        flash('Number of estimator must be a positive integer')
                        return redirect(url_for('selectAlgo'))
                    else:
                        numberEstimator = int(request.form['numberEstimator'])
                except:
                    flash('Number of estimator must be an integer')
                    return redirect(url_for('selectAlgo'))
                
                #Check whether learning rate is valid
                try:
                    if float(request.form['learningRate']) <= 0 :
                        flash('Learning rate must be a positive float')
                        return redirect(url_for('selectAlgo'))
                    else:
                        learningRate = float(request.form['learningRate'])
                except:
                    flash('Number of estimator must be an float')
                    return redirect(url_for('selectAlgo'))
                
                #Check whether loss is valid
                if not request.form['loss'] in ['linear', 'square', 'exponential']:
                    flash('Invalid loss, loss must be one of them: linear, square, exponential')
                    return redirect(url_for('selectAlgo'))
                else:
                    loss = request.form['loss']
                
                #Parameters are valid, train time
                df2 = df[selectedX+selectedY]
                df2 = dropNanAndDuplicates(df2, 0.75)
                df2, encoderArr = stringEncoder(df2, df2.loc[:, df2.dtypes == object].columns)
                trainX, testX, trainY, testY = train_test_split(df2[selectedX], df2[selectedY], 
                                                                test_size= 0.15, shuffle= True)
                model = applyAdaBoost(trainX, trainY, numberEstimator= numberEstimator, learningRate= learningRate, loss=loss)
            
                #Train is done, predict time
                result = model.predict(testX).reshape((len(testX),-1))
                if encoderArr:
                    result = stringDecoder(result, encoderArr, selectedY)
            
                return redirect(url_for('results', actual= testY, prediction= result))
            
        return render_template("select_algo.html")

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
            if request.form.getlist("parameters") != []:
                return pie_plot(df.select_dtypes(include = ["object"]),request.form.getlist("parameters"))
            else:
                return render_template('graphs/pie_plot.html',columns =  df.select_dtypes(include = ["object"]).columns,error = "Please select parameters to process.")

        return render_template('graphs/pie_plot.html',columns = df.select_dtypes(include = ["object"]).columns)

    @app.route('/dist_graph', methods = ["GET","POST"])
    def dist_graph():
        global df
        if request.method == "POST":
            print(request.form)
            if 'selected_parameter' in request.form:
                numberBin = 20 if (request.form['numberBin'].isnumeric() == False) else int(request.form['numberBin'])
                return dist_plot(df.select_dtypes(exclude = ['object']),request.form['selected_parameter'],numberBin)
            else:
                return render_template('graphs/dist_plot.html',columns =df.select_dtypes(exclude = ['object']).columns, error = "Please choose the parameter for histogram!")
        return render_template('graphs/dist_plot.html',columns = df.select_dtypes(exclude = ['object']).columns)

    @app.route('/bar_graph', methods = ["GET","POST"])
    def bar_graph():
        global df
        if request.method == "POST":
            print(request.form)
            if request.form.getlist("parameters") != []:
                if 'selected_type' in request.form:
                    selectedType = request.form["selected_type"]
                else:
                    selectedType = "Horizontal"

                return bar_plot(df.select_dtypes(include = ['object']),request.form.getlist("parameters"),selectedType)
            else:
                return render_template('graphs/bar_plot.html',columns =df.select_dtypes(include = ['object']).columns, error = "Please choose the parameter for bar graph!")
        return render_template('graphs/bar_plot.html',columns = df.select_dtypes(include = ['object']).columns)


    @app.route("/pca_transform", methods = ["GET","POST"])
    def pca_transform():
        global user_log
        if request.method == "POST":
            model_selected = df #Selected df as default, this will be changed to model selection later on.
            n_of_component = int(request.form["n_component"]) 
            print(request.form) 

            #Check if variance ratio is given
            if request.form["variance_ratio"] != "":
                variance_ratio = float(request.form["variance_ratio"])
            else:
                variance_ratio = None

            #Drop NaN-values (This will be fixed later as it causes error) and transform the data
            model_selected.dropna(axis=1,inplace=True)
            if int(n_of_component) > len(model_selected):
                n_of_component = len(model_selected)
            
            elif int(n_of_component) <= 0:
                return render_template("transformation/pca_transform.html", columns = model_selected.columns,error = "Invalid number of component! Enter a positive number.")

            new_df,pca = PCA_transformation(data = model_selected, reduce_to = n_of_component, var_ratio = variance_ratio)

            #This user-log system will be changed later on.
            user_log += [("PCA_Transformation","model_0",
            "Number of component in dataframe is reduced from {} to {}".format(model_selected.shape[1],new_df.shape[1]))]
            print(user_log)
            return PCA_transformation_describe(new_df,pca)

        return render_template("transformation/pca_transform.html")


    @app.route("/create_column", methods = ["GET","POST"])
    def create_column():
        global user_log
        if request.method == "POST":

            #Catch parameters
            selected_model = df
            selected_parameters = request.form.getlist("selected_parameters")
            selected_mode = request.form["selected_mode"]
            delete_columns = int(request.form["delete_columns"])
            new_column_name = request.form["new_column_name"]
            
            if new_column_name in selected_model.columns: # -- check if column name exist
                return render_template("transformation/create_column.html", columns = selected_model.columns,error = "This column name is already exist. Please enter a non-exist name.")
            elif new_column_name == "": # -- check if column name is not entered
                return render_template("transformation/create_column.html", columns = selected_model.columns,error = "Column name is not selected!")

            if selected_parameters == []: # -- no parameter is given
                    return render_template("transformation/create_column.html", columns = selected_model.columns,error = "Please select parameters!")
            
            new_df = combine_columns(data = selected_model, selected_columns = selected_parameters,
            new_column_name = new_column_name, mode = selected_mode, delete_column = delete_columns)
            print(new_df.shape)

            return render_template("transformation/create_column.html", columns = new_df.columns)




        return render_template("transformation/create_column.html", columns = df.columns)

    @app.route("/filter_transform", methods = ["GET","POST"])
    def filter_transform():
        if request.method == "POST":
            actions = {}
            for col in df.columns:
                    actions[col] = request.form.getlist(col)
            new_df = filter_data(df,actions)
        return render_template("transformation/filter_transform.html", cols = df.columns, objectCols = df.select_dtypes(include = "object").columns, df = df)



            
    """        
    @app.route('/getPlotCSV') # this is a job for GET, not POST
    def plot_csv():
        return send_file('outputs/Adjacency.csv',
                        mimetype='text/csv',
                        attachment_filename='Adjacency.csv',
                        as_attachment=True)
                        
    @app.route("/results", methods = ["GET","POST"])
    def results():
        global model,graph,selectedX,selectedY,selectedModel
        if request.method == "GET": 
            if model != None:
                return_result_graph(model,selectedModel) #This will return graph and upload file option
            else:
                return render_template("results.html", error = "Please train the model first!", modelExist = False)

        elif request.method == "POST":
            # check if the post request has the file part
            print(request.files)
            if 'file' not in request.files:
                #flash('No file part')
                return render_template("results.html",error = 'No file is submitted!')
            file = request.files['file']

            # check if user does not send any file
            if file.filename == '':
                #flash('No selected file')
                return render_template("results.html",error = 'No file is selected!')

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
                yExist = request.form["is-y-exist"]
                if request.form.get('is-value-type'):
                    assumption = True
                else:
                    assumption = False

                # read the file
                dataTypes, dataColumns, df = load_dataset(file_path,delimitter=delimitter,qualifier = qualifier, assumption=assumption)

                ##################################
                #Check if parameters are suitable#
                ##################################
                
                if yExist:
                    return proccess_and_show(model = model, selectedX = selectedX, selectedY = selectedY, testX = df, selectedModel = selectedModel) 
                else:
                    return proccess_and_show(model = model, selectedX = selectedX, selectedY = [], testX = df, selectedModel = selectedModel) 

            else:
                return return_result_graph(model,selectedModel)

        return "To be continued"
    """        
    init_app(app)
    return app


