import os
import numpy as np  
import pandas as pd
from flask import Flask, request, redirect, url_for,render_template,session,Response, send_file
from pandas.core.indexes.base import InvalidIndexError
from werkzeug.utils import secure_filename

from utils.auth import bp
from utils.graphs import PCA_transformation,PCA_transformation_describe,correlation_plot,pie_plot,bar_plot,scatter_matrix,dist_plot,confusion_matrix_plot
from utils.model_train import preprocess_for_model,fetch_model,train_model,model_type,test_model
from utils.transformers import apply_model_transformers,revert_model_transformers,get_metrics,combine_columns
from utils.workspace import *
from utils import *

from bokeh.resources import INLINE
from bokeh.embed import components

from sklearn.model_selection import train_test_split
import datetime


def create_app(test_config = None):
    UPLOAD_FOLDER = 'C:\\Users\\kargi\\Flask Practi e\\see\\datasets'
    ALLOWED_EXTENSIONS = set(['txt', 'csv'])
    app = Flask(__name__) 
    DATABASE=os.path.join(app.instance_path, 'flaskr.sqlite')

    app.config.from_mapping(
        SECRET_KEY='dev',
        DATABASE=os.path.join(app.instance_path, 'flaskr.sqlite'),
        UPLOAD_FOLDER = UPLOAD_FOLDER,
        FLASK_APP = "run.py",
        FLASK_ENV = "development"
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

    
    @app.errorhandler(403)
    def forbidden(e):
        return render_template('error/403.html'), 403


    @app.errorhandler(404)
    def page_not_found(e):
        return render_template('error/404.html'), 404


    @app.errorhandler(500)
    def internal_server_error(e):
        return render_template('error/500.html'), 500

    @app.route('/')
    def entry():
        if session.get('user_id'):
            return redirect(url_for('workspace'))
        return redirect(url_for('auth.login'))

    @app.route('/workspace', methods=['GET', 'POST'])
    def workspace():
    
        df = load_temp_dataframe(session.get("user_id")) #load dataframe

        if not session.get('user_log'):
            session['user_log'] = []

        if request.method == 'POST':
            
            #Add Workspace button clicked, redirect to upload screen
            if request.form.get('Add Workspace'):
                return redirect(url_for("upload_file"))


            #Delete Workspace button clicked, delete current workspace and redirect workspace page
            elif request.form.get('Delete Workspace'):
                delete_workspace(session["user_id"], session["selected_workspace"])
                session["selected_workspace"] = None
                session["selected_dataframe"] = None
                session["selected_x"] = []
                session["selected_y"] = []
                return redirect(url_for("workspace"))
            
            #Save DataFrame button clicked, save current dataframe to current workspace
            elif request.form.get('Save DataFrame'):
                source_df = 0
                desc = ""
                if df.shape[0] == 0:
                    flash("No data is selected. Please upload or select data to continue.")
                    return redirect(url_for("workspace"))
                add_checkpoint(session["user_id"], session["selected_workspace"], source_df, df, desc)
                session["selected_workspace"] = None
                session["selected_dataframe"] = None
                return redirect(url_for("workspace"))
            
            #Delete DataFrame button clicked, delete selected dataframe from workspace
            elif request.form.get('Delete DataFrame'):
                delete_checkpoint(session["user_id"], session["selected_workspace"], session["selected_dataframe"])
                session["selected_workspace"] = None
                session["selected_dataframe"] = None
                session["selected_x"] = []
                session["selected_y"] = []
                return redirect(url_for("workspace"))
            
            #Select DataFrame button clicked, change current df to session["selected_dataframe"]
            elif request.form.get('Select DataFrame'):
                df = get_checkpoint(session["user_id"], session["selected_workspace"], session["selected_dataframe"])
                save_temp_dataframe(df,session.get("user_id"))
                session["selected_workspace"] = None
                session["selected_dataframe"] = None
                session["selected_x"] = []
                session["selected_y"] = []
                return redirect(url_for("workspace"))

            elif request.form.get('Clear Log'):
                session['user_log'] = []
                return redirect(url_for("workspace"))

            elif request.form.get('Download DataFrame'):
                path = str(session.get("user_id")) + "_" + str(session.get("selected_workspace")) + "_" + str(session.get("selected_dataframe")) + ".csv"
                path = "csv/" + path 
                return redirect(url_for("download_csv",path=path))
            #Render first screen
            return render_template("workspace.html",logs=session.get('user_log')[::-1])
            
        #Determine which workspace and checkpoint is selected
        session["selected_workspace"] = request.args.get('active_workspace')
        session["selected_dataframe"] = request.args.get('active_dataframe')
        
        #Checkpoint selected, print it
        if session["selected_dataframe"] is not None:
            df2 = get_checkpoint(session["user_id"], session["selected_workspace"], session["selected_dataframe"])
            isLoaded = True
            return render_template("workspace.html", workspaces = get_workspaces(session["user_id"]), DataFrames = get_workspace(session["user_id"], session["selected_workspace"]), column_names=df2.columns.values, row_data=list(df2.head(5).values.tolist()),
                            link_column="Patient ID", zip=zip, isLoaded = isLoaded, 
                            rowS = df2.shape[0], colS = df2.shape[1], active_workspace = session["selected_workspace"],logs=session.get('user_log')[::-1])
                            
                            
        #Only workspace selected, print checkpoints of the workspace
        elif session["selected_workspace"] is not None:
            return render_template("workspace.html", workspaces = get_workspaces(session["user_id"]), DataFrames = get_workspace(session["user_id"], session["selected_workspace"]), active_workspace = session["selected_workspace"],logs=session.get('user_log')[::-1])
            
            
        #Nothing is selected, print workspaces
        else:
            return render_template("workspace.html", workspaces = get_workspaces(session["user_id"]),logs=session.get('user_log')[::-1])
            
    @app.route('/upload_file', methods=['GET', 'POST'])
    def upload_file():
        df = load_temp_dataframe(session.get("user_id"))
        if not session.get('user_log'):
            session['user_log'] = []

        if request.method == 'POST':

            df = load_(app.config["UPLOAD_FOLDER"],request.files)
            if df is not None:
                save_temp_dataframe(df,session.get("user_id"))
                session["selected_x"] = []
                session["selected_y"] = []
                create_workspace(session["user_id"], df)
                description = str(datetime.datetime.now()) + " created new workspace. "
                session["user_log"] += [description + ""] 
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
        df = load_temp_dataframe(session.get("user_id"))
        if not session.get("selected_x"):
            session["selected_x"] = []

        if not session.get('user_log'):
            session['user_log'] = []

        if request.method == 'POST':
            session["selected_x"] = request.form.getlist('hello')
            description = str(datetime.datetime.now()) + " Selected  " + str(len(session['selected_x'])) + " many variables for the model."
            session["user_log"] += [description + ""]
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
        df = load_temp_dataframe(session.get("user_id"))
        if not session.get("selected_y"):
            session["selected_y"] = []

        if not session.get("selected_x"):
            session["selected_x"] = []

        if not session.get('user_log'):
            session['user_log'] = []

        if len(session.get('selected_x')) == len(df.columns):
            flash("No possible y can be selected since all features are selected as X parameter!")
            return redirect(url_for('select_variables'))

        if request.method == 'POST':
            session["selected_y"] = request.form.getlist('hello')
            description = str(datetime.datetime.now()) + " Selected  " + str(len(session['selected_y'])) + " many target for the model."
            session["user_log"] += [description + ""]
            return redirect(url_for('selectAlgo'))

        possibleDf = df.drop(session["selected_x"],axis=1)
        
        if(len(possibleDf) != 0):
            dtypes, cols = groupColumns(possibleDf)
        else:
            dtypes = []
            cols = []
        return render_template("select_variables.html",types = dtypes, columns = cols)

    @app.route("/selectAlgo",methods = ["GET","POST"])
    def selectAlgo():
        df = load_temp_dataframe(session.get("user_id"))
        nan_found = False
        parameter_selected = True
        # check if selected_x is selected
        if not session.get("selected_x") or len(session.get("selected_x")) == 0:
            session["selected_x"] = []
            parameter_selected = False
            flash("No parameter for training is selected! Please select one to continue.")
            return redirect(url_for("select_variables"))

        # check if selected_y is selected
        if not session.get("selected_y") or len(session.get("selected_y")) == 0:
            session["selected_y"] = []
            parameter_selected = False
            flash("No parameter for prediction is selected! Please select one to continue.")
            return redirect(url_for("select_y"))

        has_NaN_value = [col for col in df.columns if df[col].isnull().any()]
        if len(has_NaN_value) > 0:
            nan_found = True
            flash("A NaN value has been detected in the dataset! Please remove them to continue.List of features that has NaN values: {}".format(has_NaN_value))

        regression_model,classification_model = model_chooser(df,session.get("selected_y"))
      
        if request.method == "POST" and nan_found == False and parameter_selected == True:
            # get the df_X and df_y
            df_X = df[session.get("selected_x")]
            df_y = df[session.get("selected_y")]

            selected_model = request.form.get("selected_model")
            session["selected_model"] = selected_model

            ##PART 1##
            selected_parameters = request.form
            df_X,df_y = preprocess_for_model(selected_model,df_X,df_y) # apply preprocess that is needed
            if model_type(selected_model) == "classification":
                one_class_columns = [col for col in df_y.columns if df_y[col].nunique() < 2]
                if len(one_class_columns) != 0:
                    flash("Some of the data we are trying to predict has only one class! Please remove such classes to continue: {}".format(one_class_columns))
                    return render_template("select_algo.html",regression_model = regression_model, 
        classification_model = classification_model,more_than_one = len(session.get('selected_y')) > 1)
            ##PART 2##
            model = fetch_model(selected_model,selected_parameters) # create model with given parameters

            ##PART 3##
            train_X, test_X, train_y, test_y = train_test_split(df_X,df_y,train_size = 0.80) # split the dataframe
            model = train_model(model,train_X,train_y) # train the model 
            session["selected_x_trained"] = session.get('selected_x')
            session["selected_y_trained"] = session.get('selected_y')
            ##PART 4##
            save_user_model(model,session.get("user_id"),body = "-user-model")
            save_temp_dataframe(pd.concat([test_X,test_y],axis = 1),session.get("user_id"), body = "-test-dataframe")
            return redirect(url_for("result"))

        return render_template("select_algo.html",regression_model = regression_model, 
        classification_model = classification_model,more_than_one = len(session.get('selected_y')) > 1)

    @app.route('/current_data', methods = ["GET","POST"])
    def current_data():
        df = load_temp_dataframe(session.get("user_id"))
        selected_model = session.get('selected_model')
        save_temp_dataframe(df,session.get('user_id'),method = "csv") # this will slow down the process
        if not session.get("selected_y"):
            session["selected_y"] = []  

        if not session.get("selected_x"):
            session["selected_x"] = []
            
        if request.method == "POST":
            number_of_head = request.form.get('head_number')
            number_of_head =  5 if check_float(number_of_head) == False else int(number_of_head) 
            return render_template("current_data.html", column_names=df.columns.values, row_data=list(df.head(number_of_head).values.tolist()),
                    link_column="Patient ID", zip=zip, rowS = df.shape[0], colS = df.shape[1],selected_model = selected_model,
                    selected_x = np.sort(session.get('selected_x')), selected_y = np.sort(session.get('selected_y')),path = "temp/" + str(session.get('user_id')) + '-df-temp' + ".csv")
        return render_template("current_data.html", column_names=df.columns.values, row_data=list(df.head(5).values.tolist()),
                            link_column="Patient ID", zip=zip, rowS = df.shape[0], colS = df.shape[1],selected_model = selected_model,
                            selected_x = np.sort(session.get('selected_x')), selected_y = np.sort(session.get('selected_y')),path = "temp/" + str(session.get('user_id')) + '-df-temp' + ".csv")


    @app.route('/result')
    def result():
        if session.get("selected_model") == None:
            flash("No model has been trained!")
            return redirect(url_for("selectAlgo"))

        # load required models/dataframes
        type_of_model = model_type(session.get("selected_model"))
        model = load_user_model(session.get('user_id'),body = "-user-model")
        test_dataframe = load_temp_dataframe(session.get('user_id'),body = "-test-dataframe")
        test_X,test_y = test_dataframe[session.get("selected_x_trained")].copy(),test_dataframe[session.get("selected_y_trained")].copy()
        predicted_y = pd.DataFrame(test_model(model,test_X),columns = test_y.columns)
        predicted_y.set_index([test_y.index],inplace=True) # match indexes

        # prepare result dataframe
        if type_of_model == "regression":
            # Revert results before calculating metrics
            _,test_y = revert_model_transformers(test_X,test_y)
            _,predicted_y = revert_model_transformers(test_X,predicted_y)
            predicted_y.columns = ["predicted_" + col for col in test_y.columns]
            test_y.columns = ["actual_" + col for col in test_y.columns]
            result_dataframe = pd.concat([test_y,predicted_y],axis=1)   
            model_scores,mse_errors,mae_errors,log_errors,f1_scores = get_metrics(type_of_model,test_y,predicted_y)
        
        else:
            # Calculate metrics before reverting results
            model_scores,mse_errors,mae_errors,log_errors,f1_scores = get_metrics(type_of_model,test_y,predicted_y)
            _,test_y = revert_model_transformers(test_X,test_y)
            _,predicted_y = revert_model_transformers(test_X,predicted_y)
            script,div = confusion_matrix_plot(test_y,predicted_y)
            predicted_y.columns = ["predicted_" + col for col in test_y.columns]
            test_y.columns = ["actual_" + col for col in test_y.columns]
            result_dataframe = pd.concat([test_y,predicted_y],axis=1)  

        save_temp_dataframe(result_dataframe,session.get('user_id'),body = "-result-dataframe", method = "csv")
        path = "temp/" + str(session.get('user_id')) + '-result-dataframe' + ".csv"

        if type_of_model == "classification":
            return render_template("result.html",model_scores = model_scores,mse_errors = mse_errors,mae_errors = mae_errors,
            f1_scores = f1_scores, log_errors = log_errors,
            column_names=result_dataframe.columns.values, row_data=list(result_dataframe.head(10).values.tolist()),
            link_column="Patient ID", zip=zip, rowS = result_dataframe.shape[0], colS = result_dataframe.shape[1],
            path = path, plot_script=script,plot_div=div,js_resources=INLINE.render_js(),css_resources=INLINE.render_css(),graphSelected = True,
            parameters = session.get("selected_y_trained"))
        else:
            return render_template("result.html",model_scores = model_scores,mse_errors = mse_errors,mae_errors = mae_errors,
            f1_scores = f1_scores, log_errors = log_errors,
            column_names=result_dataframe.columns.values, row_data=list(result_dataframe.head(10).values.tolist()),
            link_column="Patient ID", zip=zip, rowS = result_dataframe.shape[0], colS = result_dataframe.shape[1],
            path = path,graphSelected = False,parameters = session.get("selected_y_trained"))

    @app.route('/result/uploaded', methods = ["GET","POST"])
    def result_uploaded():
        if session.get("selected_model") == None:
            flash("No model has been trained!")
            return redirect(url_for("selectAlgo"))

        # load required models/dataframes
        test_dataframe = load_temp_dataframe(session.get('user_id'),body = "-test-dataframe")
        test_dataframe = test_dataframe[session.get('selected_x_trained')]
        type_of_model = model_type(session.get("selected_model"))
        model = load_user_model(session.get('user_id'),body = "-user-model")
        if request.method == "POST":
            df = load_(app.config['UPLOAD_FOLDER'],request.files)
            df.drop(session.get('selected_y_trained'), inplace = True, axis = 1)
            if df is not None:
                if check_suitable(test_dataframe,df):
                    df_X = df[session.get('selected_x_trained')]
                    df_X, _ = apply_model_transformers(df_X)
                    df_y = pd.DataFrame(model.predict(df_X),columns = session.get('selected_y_trained'))
                    
                    df_X,df_y = revert_model_transformers(df_X,df_y)
                    for col in df_y.columns:
                        df[col] = df_y[col]
                    result_dataframe = df
                    save_temp_dataframe(result_dataframe,session.get('user_id'),body = "-result-dataframe", method = "csv")
                    path = "temp/" + str(session.get('user_id')) + '-result-dataframe' + ".csv"
                    return render_template("result_user.html",
                    column_names=result_dataframe.columns.values, row_data=list(result_dataframe.head(10).values.tolist()),
                    link_column="Patient ID", zip=zip, rowS = result_dataframe.shape[0], colS =result_dataframe.shape[1],
                    path = path, parameters = session.get("selected_y_trained"), dataSelected = True)
                else:
                    flash("Data that is used is not suitable with model! Please check the parameters.")
                    return render_template("result_user.html", dataSelected = False)
        return render_template("result_user.html")
        

    @app.route('/scatter_graph', methods = ["GET","POST"])
    def scatter_graph():
        df = load_temp_dataframe(session.get("user_id"))
        if df.shape[0] == 0:
            flash("No data is selected. Please upload or select data to continue.")
            return redirect(url_for("workspace"))

        if not session.get('user_log'):
            session['user_log'] = []
        if request.method == "POST":
            if 'parameters' in request.form:
                selected_features = request.form.getlist('parameters')
            
            else:
                flash("Please select parameters to create a scatter plot.")
                return render_template('graphs/scatter_plot.html',columns = df.columns)
            if len(selected_features) >= 10:
                flash("More than 10 columns has been selected. Please select number of features between 2 and 10")
                return render_template('graphs/scatter_plot.html',columns = df.columns)

            elif len(selected_features) <= 2:
                flash("Less than 10 colums has been selected. Please select number of features between 2 and 10")
                return render_template('graphs/scatter_plot.html',columns = df.columns)
            
            else:
                description = str(datetime.datetime.now()) + " Scatter matrix is created with   " + str(len(selected_features)) +" variables."
                session["user_log"] += [description + ""]
                return scatter_matrix(df,selected_features)
        return render_template('graphs/scatter_plot.html',columns = df.columns)


    @app.route('/correlation_graph', methods = ["GET","POST"])
    def correlation_graph():
        df = load_temp_dataframe(session.get("user_id"))
        if df.shape[0] == 0:
            flash("No data is selected. Please upload or select data to continue.")
            return redirect(url_for("workspace"))

        if request.method == "POST":
            if 'parameters' in request.form:
                selected_features = request.form.getlist('parameters')
                description = str(datetime.datetime.now()) + " Correlation matrix is created with   " + str(len(selected_features)) +" variables."
                session["user_log"] += [description + ""]
                return correlation_plot(df.select_dtypes(exclude = ['object']),selected_features)
                
            else:
                flash("Please select parameters to create a correlation plot.")
                return redirect(url_for("correlation_graph"))

        return render_template('graphs/correlation_plot.html',columns =df.select_dtypes(exclude = ['object']).columns)


    @app.route('/pie_graph', methods = ["GET","POST"])
    def pie_graph():
        df = load_temp_dataframe(session.get("user_id"))
        if df.shape[0] == 0:
            flash("No data is selected. Please upload or select data to continue.")
            return redirect(url_for("workspace"))

        if request.method == "POST":
            if request.form.getlist("parameters") != []:
                if request.form.get("sort-values"):
                    sort_values = True
                else:
                    sort_values = False               
                top_values =  255 if check_float(request.form.get('head_number')) == False else int(request.form.get('head_number'))
                description = str(datetime.datetime.now()) + " Pie graph is created with   " + str(len(request.form.getlist('parameters'))) +" variables."
                session["user_log"] += [description + ""]
                return pie_plot(df.select_dtypes(include = ["object","int32","int64"]),request.form.getlist("parameters"),sort_values,top_values)

            else:
                flash("Please select parameters to process.")
                return redirect(url_for('pie_graph'))

        return render_template('graphs/pie_plot.html',columns = df.select_dtypes(include = ["object","int32","int64"]).columns)

    @app.route('/dist_graph', methods = ["GET","POST"])
    def dist_graph():
        df = load_temp_dataframe(session.get("user_id"))
        if df.shape[0] == 0:
            flash("No data is selected. Please upload or select data to continue.")
            return redirect(url_for("workspace"))

        if not session.get('user_log'):
            session['user_log'] = []
        if request.method == "POST":
            if 'selected_parameter' in request.form:
                numberBin = 20 if (request.form['numberBin'].isnumeric() == False) else int(request.form['numberBin'])
                description = str(datetime.datetime.now()) + " Distribution graph is created for   " + request.form['selected_parameter']
                session["user_log"] += [description + ""]
                return dist_plot(df.select_dtypes(exclude = ['object']),request.form['selected_parameter'],numberBin)
            else:
                flash("Please select parameters to create distribution graph.")
                return redirect(url_for("dist_graph"))
        return render_template('graphs/dist_plot.html',columns = np.sort(df.select_dtypes(exclude = ['object']).columns))

    @app.route('/bar_graph', methods = ["GET","POST"])
    def bar_graph():
        df = load_temp_dataframe(session.get("user_id"))
        if df.shape[0] == 0:
            flash("No data is selected. Please upload or select data to continue.")
            return redirect(url_for("workspace"))

        if not session.get('user_log'):
            session['user_log'] = []
        if request.method == "POST":
            if request.form.getlist("parameters") != []:
                if 'selected_type' in request.form:
                    selectedType = request.form["selected_type"]
                else:
                    selectedType = "Horizontal"
                description = str(datetime.datetime.now()) + " Bar graph is created with   " + str(len(request.form.getlist("parameters"))) +" variables."
                session["user_log"] += [description + ""]
                return bar_plot(df.select_dtypes(include = ['object']),request.form.getlist("parameters"),selectedType)
            else:
                flash("Please select parameters to create bar plot.")
                return redirect(url_for("bar_graph"))
        return render_template('graphs/bar_plot.html',columns = np.sort(df.select_dtypes(include = ['object']).columns))


    @app.route("/pca_transform", methods = ["GET","POST"])
    def pca_transform():
        df = load_temp_dataframe(session.get("user_id"))
        if df.shape[0] == 0:
            flash("No data is selected. Please upload or select data to continue.")
            return redirect(url_for("workspace"))

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
            save_temp_dataframe(df,session.get("user_id"))
            session["selected_x"] = []
            session["selected_y"] = []
            #This user-log system will be changed later on.
            description = str(datetime.datetime.now()) + " PCA transformation is applied."
            session["user_log"] += [description + ""]
            return PCA_transformation_describe(df,pca)

        return render_template("transformation/pca_transform.html")


    @app.route("/create_column", methods = ["GET","POST"])
    def create_column():
        df = load_temp_dataframe(session.get("user_id"))
        if df.shape[0] == 0:
            flash("No data is selected. Please upload or select data to continue.")
            return redirect(url_for("workspace"))

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
            
            if new_column_name in df.columns and new_column_name is not "" : # -- check if column name exist
                flash("This column name is already exist!")
                return redirect("create_column")

            elif (new_column_name == None or new_column_name == "") and (selected_mode in ["sum","mean","difference","concat"]): # -- check if column name is not entered
                flash("Please enter a column name!")
                return redirect("create_column")

            if selected_parameters == []: # -- no parameter is given
                flash("Please select parameters!")
                return redirect("create_column")
            
            df,transformer = combine_columns(data = df, selected_columns = selected_parameters,
            new_column_name = new_column_name, mode = selected_mode, delete_column = delete_columns)
            save_temp_dataframe(df,session.get("user_id"))
            session["selected_x"] = []
            session["selected_y"] = []
            description = str(datetime.datetime.now()) + " New column is created."
            session["user_log"] += [description + ""]

        return render_template("transformation/create_column.html", columns = df.columns)

    @app.route("/filter_transform", methods = ["GET","POST"])
    def filter_transform():
        df = load_temp_dataframe(session.get("user_id"))
        if df.shape[0] == 0:
            flash("No data is selected. Please upload or select data to continue.")
            return redirect(url_for("workspace"))

        if not session.get('user_log'):
            session['user_log'] = []
        if request.method == "POST":
            actions = {}
            for col in df.columns: # remove colums with empty data
                col_list = remove_empty_lists(request.form.getlist(col))
                if col_list:
                    actions[col] = col_list
            df = filter_data(df,actions)
            save_temp_dataframe(df,session.get("user_id"))
            session["selected_x"] = []
            session["selected_y"] = []
            description = str(datetime.datetime.now()) + " Parameters are filtered." 
            session["user_log"] += [description + ""]
        return render_template("transformation/filter_transform.html", cols = df.columns, objectCols = df.select_dtypes(include = "object").columns, df = df)

    @app.route('/download_csv')
    def download_csv():
        path = request.args.get('path')
        if path is None:
            flash("An error occured during downloading file!")
            return redirect(url_for('workspace'))
        return send_file(path,mimetype="text/csv",attachment_filename='myfile.csv',as_attachment=True)
    

    app.register_blueprint(bp, url_prefix='/auth')
    init_app(app)
    return app



