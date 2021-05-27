import os
import numpy as np  
import pandas as pd
from bokeh.resources import INLINE
from flask import Flask, request, redirect, url_for,render_template,session,Response, send_file
from werkzeug.utils import secure_filename
from utils import *
from modelTrain import *
from preprocess import *
from auth import *
from db import *
from workspace import *
from sklearn.model_selection import train_test_split
import datetime


### These variables will be fixed later on as they are global and will cause errors. ###
df = pd.DataFrame()
graph = None
selectedModel = None
########################################################################################


def create_app(test_config = None):
    UPLOAD_FOLDER = 'C:\\Users\\kargi\\Flask Practi e\\see\\datasets'
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

    
    @app.route('/')
    def entry():
        if session.get('user_id'):
            return redirect(url_for('workspace'))
        return redirect(url_for('auth.login'))

    @app.route('/workspace', methods=['GET', 'POST'])
    def workspace():
        global df
        print(session.get('user_id'))
        if not session.get("user_id"):
            session["user_id"] = 0

        if not session.get('user_log'):
            session['user_log'] = []

        print("User log is : ",session.get('user_log'))
        if request.method == 'POST':
            
            #Add Workspace button clicked, redirect to upload screen
            if request.form.get('Add Workspace'):
                return redirect(url_for("upload_file"))

              
            #Delete Workspace button clicked, delete current workspace and redirect workspace page
            elif request.form.get('Delete Workspace'):
                delete_workspace(session["user_id"], session["selected_workspace"])
                session["selected_workspace"] = None
                session["selected_dataframe"] = None
                return redirect(url_for("workspace"))
            
            #Save DataFrame button clicked, save current dataframe to current workspace
            elif request.form.get('Save DataFrame'):
                source_df = 0
                desc = ""
                add_checkpoint(session["user_id"], session["selected_workspace"], source_df, df, desc)
                session["selected_workspace"] = None
                session["selected_dataframe"] = None
                return redirect(url_for("workspace"))
            
            #Delete DataFrame button clicked, delete selected dataframe from workspace
            elif request.form.get('Delete DataFrame'):
                delete_checkpoint(session["user_id"], session["selected_workspace"], session["selected_dataframe"])
                session["selected_workspace"] = None
                session["selected_dataframe"] = None
                return redirect(url_for("workspace"))
            
            #Select DataFrame button clicked, change current df to session["selected_dataframe"]
            elif request.form.get('Select DataFrame'):
                df = get_checkpoint(session["user_id"], session["selected_workspace"], session["selected_dataframe"])
                session["selected_workspace"] = None
                session["selected_dataframe"] = None
                return redirect(url_for("workspace"))

            elif request.form.get('Clear Log'):
                session['user_log'] = []
                return redirect(url_for("workspace"))
            #Render first screen
            return render_template("workspace.html",logs=session.get('user_log'))
            
            
        #Determine which workspace and checkpoint is selected
        session["selected_workspace"] = request.args.get('active_workspace')
        session["selected_dataframe"] = request.args.get('active_dataframe')
        print("Selected workspace and dataframe is :",session["selected_workspace"],session["selected_dataframe"])
        
        #Checkpoint selected, print it
        if session["selected_dataframe"] is not None:
            df2 = get_checkpoint(session["user_id"], session["selected_workspace"], session["selected_dataframe"])
            print("DF2 is : ", session["user_id"],session["selected_workspace"],session["selected_dataframe"])
            isLoaded = True
            return render_template("workspace.html", workspaces = get_workspaces(session["user_id"]), DataFrames = get_workspace(session["user_id"], session["selected_workspace"]), column_names=df2.columns.values, row_data=list(df2.head(5).values.tolist()),
                            link_column="Patient ID", zip=zip, isLoaded = isLoaded, 
                            rowS = df2.shape[0], colS = df2.shape[1], active_workspace = session["selected_workspace"],logs=session.get('user_log'))
                            
                            
        #Only workspace selected, print checkpoints of the workspace
        elif session["selected_workspace"] is not None:
            print(get_workspace(session["user_id"],session["selected_workspace"]),session["selected_workspace"])
            return render_template("workspace.html", workspaces = get_workspaces(session["user_id"]), DataFrames = get_workspace(session["user_id"], session["selected_workspace"]), active_workspace = session["selected_workspace"],logs=session.get('user_log'))
            
            
        #Nothing is selected, print workspaces
        else:
            return render_template("workspace.html", workspaces = get_workspaces(session["user_id"]),logs=session.get('user_log'))
            
    @app.route('/upload_file', methods=['GET', 'POST'])
    def upload_file():
        global df
        if not session.get('user_log'):
            session['user_log'] = []

        if request.method == 'POST':

            # check if the post request has the file part
            print(request.files)
            if 'file' not in request.files:
                flash('No file part')
                return redirect(url_for("upload_file"))
            file = request.files['file']

            # check if user does not send any file
            if file.filename == '':
                flash('No selected file')
                return redirect(url_for("upload_file"))

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
                _, _, df = load_dataset(file_path,delimitter=delimitter,qualifier = qualifier, assumption=assumption)
                create_workspace(session["user_id"], df)
                
                description = str(session["user_id"]) + " created new workspace at ",str(datetime.datetime.now())
                session["user_log"] += [description]
                isLoaded = True
                return render_template("upload_file.html", column_names=df.columns.values, row_data=list(df.head(5).values.tolist()),
                            link_column="Patient ID", zip=zip, isLoaded = isLoaded, rowS = df.shape[0], colS = df.shape[1])
            else:
                flash("Extension is not correct !")
                return redirect(url_for("upload_file"))
        else:                        
            return render_template("upload_file.html")

    #Select x-variables among checkboxes
    @app.route("/select_variables", methods = ["GET","POST"])
    def select_variables():
        global df
        if not session.get("selected_x"):
            session["selected_x"] = []

        if not session.get('user_log'):
            session['user_log'] = []

        if request.method == 'POST':
            session["selected_x"] = request.form.getlist('hello')
            description = str(datetime.datetime.now()), " Selected  " + str(len(session['selected_x'])) + " many variables for the model."
            session["user_log"] += [description]
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
        global df
        if not session.get("selected_y"):
            session["selected_y"] = []

        if not session.get("selected_x"):
            session["selected_x"] = []

        if not session.get('user_log'):
            session['user_log'] = []

        if request.method == 'POST':
            session["selected_y"] = request.form.getlist('hello')
            description = str(datetime.datetime.now()), " Selected  " + str(len(session['selected_y'])) + " many target for the model."
            session["user_log"] += [description]
            return redirect(url_for('selectAlgo'))

        possibleDf = df.drop(session["selected_x"],axis=1)
        
        if(len(possibleDf) != 0):
            dtypes, cols = groupColumns(possibleDf)
        else:
            dtypes = []
            cols = []
        return render_template("select_variables.html",types = dtypes, columns = cols)


    @app.route("/selectAlgo", methods = ["GET","POST"])
    def selectAlgo():
        global df
        if not session.get("selected_x"):
            session["selected_x"] = []

        if not session.get("selected_y"):
            session["selected_y"] = []

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
                df2 = df[session["selected_x"]+session["selected_y"]]
                df2 = dropNanAndDuplicates(df2, 0.75)
                df2, encoderArr = stringEncoder(df2, df2.loc[:, df2.dtypes == object].columns)
                df2, scaler, scalerY = scale(df2, session["selected_y"])
                trainX, testX, trainY, testY = train_test_split(df2[session["selected_x"]], df2[session["selected_y"]], 
                                                                test_size= 0.15, shuffle= True)
            
                session["model"] = applySVM(trainX, trainY, kernel= kernel, c= float(C), gamma= gamma, degree= float(degree))
            
                #Train is done, predict time
                result = session["model"].predict(testX).reshape((len(testX),-1))
                result = scalerY.inverse_transform(result)
                if encoderArr:
                    result = stringDecoder(result, encoderArr, session["selected_y"])
            
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
                df2 = df[session["selected_x"]+session["selected_y"]]
                df2 = dropNanAndDuplicates(df2, 0.75)
                df2, encoderArr = stringEncoder(df2, df2.loc[:, df2.dtypes == object].columns)
                trainX, testX, trainY, testY = train_test_split(df2[session["selected_x"]], df2[session["selected_y"]], 
                                                                test_size= 0.15, shuffle= True)
                session["model"] = applyRandomForest(trainX, trainY, numberEstimator= numberEstimator, maxDepth = maxDepth, minSamplesLeaf=minSamplesLeaf)
            
                #Train is done, predict time
                result = session["model"].predict(testX).reshape((len(testX),-1))
                if encoderArr:
                    result = stringDecoder(result, encoderArr, session["selected_y"])
                    
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
                df2 = df[session["selected_x"]+session["selected_y"]]
                df2 = dropNanAndDuplicates(df2, 0.75)
                df2, encoderArr = stringEncoder(df2, df2.loc[:, df2.dtypes == object].columns)
                trainX, testX, trainY, testY = train_test_split(df2[session["selected_x"]], df2[session["selected_y"]], 
                                                                test_size= 0.15, shuffle= True)
                session["model"] = applyAdaBoost(trainX, trainY, numberEstimator= numberEstimator, learningRate= learningRate, loss=loss)
            
                #Train is done, predict time
                result = session["model"].predict(testX).reshape((len(testX),-1))
                if encoderArr:
                    result = stringDecoder(result, encoderArr, session["selected_y"])
            
                return redirect(url_for('results', actual= testY, prediction= result))
            
        return render_template("select_algo.html")

        
    @app.route('/scatter_graph', methods = ["GET","POST"])
    def scatter_graph():
        global df

        if not session.get('user_log'):
            session['user_log'] = []
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
                description = str(datetime.datetime.now()), " Scatter matrix is created with   " + str(len(selected_features)) +" variables."
                session["user_log"] += [description]
                return scatter_matrix(df,selected_features)
        return render_template('graphs/scatter_plot.html',columns = df.columns)


    @app.route('/correlation_graph', methods = ["GET","POST"])
    def correlation_graph():
        global df
        if request.method == "POST":
            if 'parameters' in request.form:
                selected_features = request.form.getlist('parameters')
                description = str(datetime.datetime.now()), " Correlation matrix is created with   " + str(len(selected_features)) +" variables."
                session["user_log"] += [description]
                return correlation_plot(df.select_dtypes(exclude = ['object']),selected_features)
                
            else:
                return render_template('graphs/correlation_plot.html',columns = df.select_dtypes(exclude = ['object']).columns,error = "Please select parameters to process.")

        return render_template('graphs/correlation_plot.html',columns =df.select_dtypes(exclude = ['object']).columns)


    @app.route('/pie_graph', methods = ["GET","POST"])
    def pie_graph():
        global df
        if request.method == "POST":
            if request.form.getlist("parameters") != []:
                if request.form.get("sort-values"):
                    sort_values = True
                else:
                    sort_values = False

                description = str(datetime.datetime.now()), " Pie graph is created with   " + str(len(request.form.getlist('parameters'))) +" variables."
                session["user_log"] += [description]
                return pie_plot(df.select_dtypes(include = ["object"]),request.form.getlist("parameters"),sort_values)

            else:
                flash("Please select parameters to process.")
                return redirect('pie_plot')

        return render_template('graphs/pie_plot.html',columns = df.select_dtypes(include = ["object"]).columns)

    @app.route('/dist_graph', methods = ["GET","POST"])
    def dist_graph():
        global df

        if not session.get('user_log'):
            session['user_log'] = []
        if request.method == "POST":
            print(request.form)
            if 'selected_parameter' in request.form:
                numberBin = 20 if (request.form['numberBin'].isnumeric() == False) else int(request.form['numberBin'])
                description = str(datetime.datetime.now()), " Scatter matrix is created for   " + request.form['selected_parameter']
                session["user_log"] += [description]
                return dist_plot(df.select_dtypes(exclude = ['object']),request.form['selected_parameter'],numberBin)
            else:
                return render_template('graphs/dist_plot.html',columns =df.select_dtypes(exclude = ['object']).columns, error = "Please choose the parameter for histogram!")
        return render_template('graphs/dist_plot.html',columns = df.select_dtypes(exclude = ['object']).columns)

    @app.route('/bar_graph', methods = ["GET","POST"])
    def bar_graph():
        global df
        if not session.get('user_log'):
            session['user_log'] = []
        if request.method == "POST":
            print(request.form)
            if request.form.getlist("parameters") != []:
                if 'selected_type' in request.form:
                    selectedType = request.form["selected_type"]
                else:
                    selectedType = "Horizontal"
                description = str(datetime.datetime.now()), " Scatter matrix is created with   " + str(len(request.form.getlist("parameters"))) +" variables."
                session["user_log"] += [description]
                return bar_plot(df.select_dtypes(include = ['object']),request.form.getlist("parameters"),selectedType)
            else:
                return render_template('graphs/bar_plot.html',columns =df.select_dtypes(include = ['object']).columns)
        return render_template('graphs/bar_plot.html',columns = df.select_dtypes(include = ['object']).columns)


    @app.route("/pca_transform", methods = ["GET","POST"])
    def pca_transform():
        global df

        if not session.get('user_log'):
            session['user_log'] = []
        if request.method == "POST":
            
            if request.form["n_component"].isnumeric():
                n_of_component = int(request.form["n_component"]) 
                if int(n_of_component) > len(df):
                    n_of_component = len(df)
            
                elif int(n_of_component) <= 0:
                    flash("Invalid number of component! Enter a positive number.")
                    return redirect("pca_transform")

            else:
                n_of_component = None

            #Check if variance ratio is given
            if request.form["variance_ratio"] != "":
                variance_ratio = float(request.form["variance_ratio"])
            else:
                variance_ratio = None

            if df.isnull().sum().sum() != 0:
                flash("NaN value detected!")
                return redirect("pca_transform")


            df,pca = PCA_transformation(data = df, reduce_to = n_of_component, var_ratio = variance_ratio)
            #This user-log system will be changed later on.
            description = str(datetime.datetime.now()), " PCA transformation is applied."
            session["user_log"] += [description]
            return PCA_transformation_describe(df,pca)

        return render_template("transformation/pca_transform.html")


    @app.route("/create_column", methods = ["GET","POST"])
    def create_column():
        global df

        if not session.get('user_log'):
            session['user_log'] = []
        if request.method == "POST":

            #Catch parameters
            selected_parameters = request.form.getlist("selected_parameters")
            selected_mode = request.form["selected_mode"]
            if request.form.get("delete_columns"):
                delete_columns = True
            else:
                delete_columns = False
            new_column_name = request.form.get("new_column_name")
            
            if new_column_name in df.columns: # -- check if column name exist
                flash("This column name is already exist!")
                return redirect("create_column")

            elif (new_column_name == None or new_column_name == "") and (selected_mode != "drop-nan-rows" and selected_mode != "drop-nan-columns"): # -- check if column name is not entered
                flash("Please enter a column name!")
                return redirect("create_column")

            if selected_parameters == []: # -- no parameter is given
                flash("Please select parameters!")
                return redirect("create_column")
            
            df = combine_columns(data = df, selected_columns = selected_parameters,
            new_column_name = new_column_name, mode = selected_mode, delete_column = delete_columns)
            description = str(datetime.datetime.now()), " New column is created."
            session["user_log"] += [description]

        return render_template("transformation/create_column.html", columns = df.columns)

    @app.route("/filter_transform", methods = ["GET","POST"])
    def filter_transform():
        global df
        if not session.get('user_log'):
            session['user_log'] = []
        if request.method == "POST":
            actions = {}
            for col in df.columns:
                    actions[col] = request.form.getlist(col)
            df = filter_data(df,actions).to_dict('list')
            description = str(datetime.datetime.now()), " Parameters are filtered." 
            session["user_log"] += [description]
        return render_template("transformation/filter_transform.html", cols = df.columns, objectCols = df.select_dtypes(include = "object").columns, df = df)

    @app.route('/download_csv')
    def download_csv():
        path = request.args.get('path')
        print(path)
        if path is None:
            flash("An error occured during downloading csv!")
            return redirect(url_for('workspace'))
        return send_file(path,mimetype="text/csv",attachment_filename='mygraph.csv',as_attachment=True)
    

    @app.route('/result',methods=["GET","POST"])
    def result():
        return "SIKE"
            
    """        
    @app.route('/getPlotCSV') # this is a job for GET, not POST
    def plot_csv():
        return send_file('outputs/Adjacency.csv',
                        mimetype='text/csv',
                        attachment_filename='Adjacency.csv',
                        as_attachment=True)
                        
    @app.route("/results", methods = ["GET","POST"])
    def results():
        global session["model"],graph,session["selected_x"],session["selected_y"],selectedModel
        if request.method == "GET": 
            if session["model"] != None:
                return_result_graph(session["model"],selectedModel) #This will return graph and upload file option
            else:
                return render_template("results.html", error = "Please train the session["model"] first!", modelExist = False)

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
                    return proccess_and_show(session["model"] = session["model"], session["selected_x"] = session["selected_x"], session["selected_y"] = session["selected_y"], testX = df, selectedModel = selectedModel) 
                else:
                    return proccess_and_show(session["model"] = session["model"], session["selected_x"] = session["selected_x"], session["selected_y"] = [], testX = df, selectedModel = selectedModel) 

            else:
                return return_result_graph(session["model"],selectedModel)

        return "To be continued"
    """        

    app.register_blueprint(bp, url_prefix='/auth')
    init_app(app)
    return app


