import os
import numpy as np  
import pandas as pd
from flask import Flask, request, redirect, url_for,render_template,session,Response, send_file
from pandas.core.indexes.base import InvalidIndexError
from tensorflow.python.ops.gen_math_ops import sqrt
from werkzeug.utils import secure_filename

from utils.auth import bp
from utils.graphs import PCA_transformation,PCA_transformation_describe,correlation_plot,pie_plot,bar_plot,scatter_matrix,dist_plot,confusion_matrix_plot
from utils.model_train import create_DNN, preprocess_for_model,fetch_model, train_DNN,train_model,model_type,test_model,cross_validate_models
from utils.transformers import apply_model_transformers,revert_model_transformers,get_metrics,combine_columns
from utils.workspace import *
from utils import *

from bokeh.resources import INLINE
from bokeh.embed import components

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import datetime


def create_app(test_config = None):
    """
    Create a Flask application with a function. Return application at the end of the function. 
    
    Parameters
    :test_config: (dict) -- if given, construct the model with this parameters configurations
    """
    if os.path.isdir("datasets") == False: # if file does not exist, create instead
        os.makedirs("datasets")
    UPLOAD_FOLDER = 'datasets'
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

    
    @app.errorhandler(403)
    def forbidden(e):
        """
        Render an error page if the status code is 403 (Forbiden)
        """
        return render_template('error/403.html'), 403


    @app.errorhandler(404)
    def page_not_found(e):
        """
        Render an error page if the status code is 404 (Page Not Found)
        """
        return render_template('error/404.html'), 404


    @app.errorhandler(500)
    def internal_server_error(e):
        """
        Render an error page if the status code is 500 (Internal Server Error)
        """
        return render_template('error/500.html'), 500

    @app.route('/')
    def entry():
        """
        Root page. Make sure that user is registered or logged in.
        """
        if session.get('user_id'):
            return redirect(url_for('workspace'))
        return redirect(url_for('auth.login'))

    @app.route('/workspace', methods=['GET', 'POST'])
    def workspace():
        """
        Workspace page for the website. User can interact with workspaces by
            1) Add Workspace
            2) Delete Workspace
        sections. In a workspace, user can:
            1) Save current data into current workspace
            2) Load data into current data
            3) Delete data
        """
    
        df = load_temp_dataframe(session.get("user_id")) # load current dataframe

        if not session.get('user_log'):
            session['user_log'] = []

        if request.method == 'POST':
            
            #Add Workspace button clicked, redirect to upload screen
            if request.form.get('Add Workspace'):
                session["assigned_types"] = {}
                session["assigned_columns"] = {}
                return redirect(url_for("upload_file"))


            #Delete Workspace button clicked, delete current workspace and redirect workspace page
            elif request.form.get('Delete Workspace'):
                delete_workspace(session["user_id"], session["selected_workspace"])
                description = "[ " + str(datetime.datetime.now()) + " ] >>" + " Deleted workspace. (Workspace id:{}) ".format(session.get("selected_workspace"))
                session["user_log"] += [description + ""] 

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

                description = "[ " + str(datetime.datetime.now()) + " ] >>" + " Saved the current data to (Workspace id:{}) ".format(session.get("selected_workspace"))
                session["user_log"] += [description + ""] 

                session["selected_workspace"] = None
                session["selected_dataframe"] = None


                return redirect(url_for("workspace"))
            
            #Delete DataFrame button clicked, delete selected dataframe from workspace
            elif request.form.get('Delete DataFrame'):
                delete_checkpoint(session["user_id"], session["selected_workspace"], session["selected_dataframe"])

                description = "[ " + str(datetime.datetime.now()) + " ] >>" + " Deleted dataframe. (Workspace id:{}, Data id:{}) ".format(session.get("selected_workspace"),
                session.get("selected_dataframe"))
                session["user_log"] += [description + ""] 

                session["selected_workspace"] = None
                session["selected_dataframe"] = None
                session["selected_x"] = []
                session["selected_y"] = []
                return redirect(url_for("workspace"))
            
            #Select DataFrame button clicked, change current df to session["selected_dataframe"]
            elif request.form.get('Select DataFrame'):
                df = get_checkpoint(session["user_id"], session["selected_workspace"], session["selected_dataframe"])
                save_temp_dataframe(df,session.get("user_id"))

                description = "[ " + str(datetime.datetime.now()) + " ] >>" + " A new data is selected. (Workspace id:{}, Data id:{}) ".format(session.get("selected_workspace"),
                session.get("selected_dataframe"))
                session["user_log"] += [description + ""] 

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
                path = os.path.join("csv",path)
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
    

    @app.route('/assign_types', methods = ["GET","POST"])
    def assign_unknown_types():
        if len(session.get("unknown_types")) == 0 and session.get("unknown_columns") == 0:
            flash("Everything is correct!")
            return redirect(url_for("workspace"))

        if session.get("unknown_types") is None: # types are correct
            session["unknown_types"] = []

        if session.get("unknown_columns") is None: # values are correct
            session["unknown_columns"] = []  

        if session.get('referrer_page') is None: # redirect if referrer_page is unknown
            return redirect(url_for("workspace"))

        DATATYPES_SET = ["bool","datetime64","float64","int64","object"] # avaible dtypes for pandas
        if request.method == "POST":
            assigned_types = session["assigned_types"]
            assigned_columns = session["assigned_columns"]
            
            for unknown_type in session.get("unknown_types"): # assign datatypes
                assigned_type = request.form.get(unknown_type)
                assigned_types[unknown_type] = assigned_type
    
            for unknown_column in session.get("unknown_columns"): # assign datatypes
                assigned_column = request.form.get(unknown_column)
                assigned_columns[unknown_column] = assigned_column

            session["assigned_types"] = assigned_types # assigned successfully
            session["assigned_columns"] = assigned_columns
            session["unknown_types"] = [] # reset unknown_types
            session["unknown_columns"] = []
            flash("Unknown types assigned successfully!")

            return redirect(url_for(session.get('referrer_page')))

        return render_template("assign_types.html", unknown_types = session.get("unknown_types"),
        unknown_columns = session.get("unknown_columns"),dataset = DATATYPES_SET)

    @app.route('/upload_file', methods=['GET', 'POST'])
    def upload_file():
        """
        Page where user can upload their local file into a workspace. Note that uploading a file also changes
        the current dataframe user is using. User can either upload,
            1) CSV file
            2) TXT file which is outputted from an Excel file
        """

        session["unknown_types"] = []
        session["unknown_columns"] = []
        session['referrer_page'] = 'upload_file'

        df = load_temp_dataframe(session.get("user_id"))
        if not session.get('user_log'):
            session['user_log'] = []

        if request.method == 'POST':

            df,unknown_types,unknown_columns = load_(app.config["UPLOAD_FOLDER"],request.files) # load the uploaded DataFrame. 
            if len(unknown_types) != 0 or len(unknown_columns) != 0:
                return redirect(url_for("assign_unknown_types"))

            if df is not None: # loaded successfully
                save_temp_dataframe(df,session.get("user_id"))
                
                # Make sure that parameters selected for DataFrame is reset. 
                session["selected_x"] = []
                session["selected_y"] = []
                
                last_workspace_id = create_workspace(session["user_id"], df) # Create new workspace for newly uploaded file.
                description = "[ " + str(datetime.datetime.now()) + " ] >> " + " Created new workspace. (Workspace id:{}) ".format(last_workspace_id)
                session["user_log"] += [description + ""] 
                isLoaded = True
                return render_template("upload_file.html", column_names=df.columns.values, row_data=list(df.head(5).values.tolist()),
                            link_column="Patient ID", zip=zip, isLoaded = isLoaded, rowS = df.shape[0], colS = df.shape[1])
            else: # loaded unsucessfully
                flash("Extension is not correct !")
                return redirect(url_for("upload_file"))
        else:                        
            return render_template("upload_file.html")

    @app.route("/select_variables", methods = ["GET","POST"])
    def select_variables():
        """
        Select features that will be used in training the model. Features selected in this page
        will be used to predict the parameters selected in the "select_y" page. 
        """
        df = load_temp_dataframe(session.get("user_id"))
        if not session.get("selected_x"):
            session["selected_x"] = []

        if not session.get('user_log'):
            session['user_log'] = []

        if request.method == 'POST':
            session["selected_x"] = request.form.getlist('hello')
            description = "[ " + str(datetime.datetime.now()) + " ] >>" + " Selected  " + str(len(session['selected_x'])) + " many attributes for the model."
            session["user_log"] += [description + ""]
            return redirect(url_for('select_y'))

        if(len(df) != 0):
            dtypes, cols = groupColumns(df) # to seperate dtypes and their respective columns 
        else:
            dtypes = []
            cols = []
        return render_template("select_variables.html",types = dtypes, columns = cols)

    @app.route("/select_y",methods = ["GET","POST"])
    def select_y():
        """
        Select features that will be predicted by the model. Features selected in this page will be used by model to guide the
        performance of the model.
        """
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
            description = "[ " + str(datetime.datetime.now()) + " ] >>" + " Selected  " + str(len(session['selected_y'])) + " many decision variables."
            session["user_log"] += [description + ""]
            return redirect(url_for('selectAlgo'))

        possibleDf = df.drop(session["selected_x"],axis=1) # from all possible columns, drop the columns we have selected in the select_x page
        
        if(len(possibleDf) != 0):
            dtypes, cols = groupColumns(possibleDf) # to seperate dtypes and their respective columns 
        else:
            dtypes = []
            cols = []
        return render_template("select_y_variable.html",types = dtypes, columns = cols)

    @app.route('/current_data', methods = ["GET","POST"])
    def current_data():
        """
        Show the dataframe currently being used. 
        """
        df = load_temp_dataframe(session.get("user_id"))
        selected_model = session.get('selected_model')
        save_temp_dataframe(df,session.get('user_id'),method = "csv") # this will slow down the process, but is required for downloading the data
        if not session.get("selected_y"):
            session["selected_y"] = []  

        if not session.get("selected_x"):
            session["selected_x"] = []
            
        if request.method == "POST":
            number_of_head = request.form.get('head_number')
            number_of_head =  5 if check_float(number_of_head) == False else int(number_of_head) # number of rows to be displayed for the data
            return render_template("current_data.html", column_names=df.columns.values, row_data=list(df.head(number_of_head).values.tolist()),
                    link_column="Patient ID", zip=zip, rowS = df.shape[0], colS = df.shape[1],selected_model = selected_model,
                    selected_x = np.sort(session.get('selected_x')), selected_y = np.sort(session.get('selected_y')),path = os.path.join("temp",str(session.get('user_id')) + '-df-temp' + ".csv"))
        
        return render_template("current_data.html", column_names=df.columns.values, row_data=list(df.head(5).values.tolist()),
                            link_column="Patient ID", zip=zip, rowS = df.shape[0], colS = df.shape[1],selected_model = selected_model,
                            selected_x = np.sort(session.get('selected_x')), selected_y = np.sort(session.get('selected_y')),path = os.path.join("temp",str(session.get('user_id')) + '-df-temp' + ".csv"))


    @app.route("/selectAlgo",methods = ["GET","POST"])
    def selectAlgo():
        """
        Select the algorithm/model user will use. Make sure that parameters are selected and there is no NaN value
        in the data - as it will hinder the training of our model.
        """
        df = load_temp_dataframe(session.get("user_id"))
        nan_found = False
        parameter_selected = True
        
        # check if selected_x is selected, if it is not, flash a warning
        if not session.get("selected_x") or len(session.get("selected_x")) == 0:
            session["selected_x"] = []
            parameter_selected = False
            flash("No parameter for training is selected! Please select one to continue.")
            return redirect(url_for("select_variables"))

        # check if selected_y is selected, if it is not, flash a warning
        if not session.get("selected_y") or len(session.get("selected_y")) == 0:
            session["selected_y"] = []
            parameter_selected = False
            flash("No parameter for prediction is selected! Please select one to continue.")
            return redirect(url_for("select_y"))

        # check the NaN values, if found any, flash a warning.
        has_NaN_value = [col for col in df.columns if df[col].isnull().any()]
        if len(has_NaN_value) > 0:
            nan_found = True
            flash("A NaN value has been detected in the dataset! Please remove them to continue.List of features that has NaN values: {}".format(has_NaN_value))

        regression_model,classification_model = model_chooser(df,session.get("selected_y")) 

        if request.method == "POST" and nan_found == False and parameter_selected == True: # everything is fine, create and train the model
            # get the X and y part of the data
            df_X = df[session.get("selected_x")]
            df_y = df[session.get("selected_y")]

            selected_model = request.form.get("selected_model")
            session["selected_model"] = selected_model

            # preprocess the data
            selected_parameters = request.form
            df_X,df_y = preprocess_for_model(selected_model,df_X,df_y) # apply preprocess that is needed for particular model
            print("Preprocess success!",df_y.head())
            if model_type(selected_model) == "classification": # task is a classification task
                one_class_columns = [col for col in df_y.columns if df_y[col].nunique() < 2]
                if len(one_class_columns) != 0: # if there is only one value for a discrete variable that will be predicted, flash a warning
                    flash("Some of the data we are trying to predict has only one class! Please remove such classes to continue: {}".format(one_class_columns))
                    return render_template("select_algo.html",regression_model = regression_model,
                                           classification_model = classification_model,more_than_one = len(session.get('selected_y')) > 1)
            
            # create the model
            model = fetch_model(selected_model,selected_parameters) # create model with given parameters

            K = 5 if check_float(selected_parameters.get("K")) == False or selected_parameters.get("K") == "" else int(selected_parameters.get("K"))
            
            model_scores,models,(test_y_rescaled,pred_y_rescaled) = cross_validate_models(df_X,df_y,model,K)
            if len(models) == 0:
                flash("An error occured while constructing the model!")
                return(redirect(url_for("selectAlgo")))

            best_model = models[-1] # best model is chosen randomly
            session["selected_x_trained"] = session.get('selected_x')
            session["selected_y_trained"] = session.get('selected_y')
            session["model_scores"] = model_scores  # a small price 
            session["k_cross_val"] = K

            # convert arrays to df and concat results
            save_temp_dataframe(pd.concat([test_y_rescaled,pred_y_rescaled],axis = 1),session.get("user_id"), body = "-test-dataframe") 
            save_user_model(best_model,session.get("user_id"),body = "-user-model")

            description = "[ " + str(datetime.datetime.now()) + " ] >>" + " Created new model (Model : {}). ".format(session.get("selected_model"))
            session["user_log"] += [description + ""] 
            return redirect(url_for("result"))

        return render_template("select_algo.html",regression_model = regression_model, 
        classification_model = classification_model,more_than_one = len(session.get('selected_y')) > 1)

    @app.route("/selectAlgo/DL", methods = ["GET","POST"])
    def selectAlgo_DL():
        """
        Select the algorithm/model user will use. Make sure that parameters are selected and there is no NaN value
        in the data - as it will hinder the training of our model.
        """
        
        df = load_temp_dataframe(session.get("user_id"))
        nan_found = False # true if nan values detected
        parameter_selected = True # x and y chosen correctly
        no_of_unique = None # will be used in classification tasks
        input_shape = (None,None,None) # not required for non-image inputs 
        
        # check if selected_x is selected, if it is not, flash a warning
        if not session.get("selected_x") or len(session.get("selected_x")) == 0:
            session["selected_x"] = []
            parameter_selected = False
            flash("No parameter for training is selected! Please select one to continue.")
            return redirect(url_for("select_variables"))

        # check if selected_y is selected, if it is not, flash a warning
        if not session.get("selected_y") or len(session.get("selected_y")) == 0:
            session["selected_y"] = []
            parameter_selected = False
            flash("No parameter for prediction is selected! Please select one to continue.")
            return redirect(url_for("select_y"))

        # check the NaN values, if found any, flash a warning.
        has_NaN_value = [col for col in df.columns if df[col].isnull().any()]
        if len(has_NaN_value) > 0:
            nan_found = True
            flash("A NaN value has been detected in the dataset! Please remove them to continue.List of features that has NaN values: {}".format(has_NaN_value))

        regression_model,classification_model = model_chooser(df,session.get("selected_y")) 
        if request.method == "POST" and nan_found == False and parameter_selected == True: 
            
            # get the X and y part of the data
            df_X = df[session.get("selected_x")]
            df_y = df[session.get("selected_y")]

            # determinde the model
            if request.form.get("is_image") == "yes": # if data is image
                selected_model = "CNN"
            else:
                selected_model = "DNN"

            if regression_model and not classification_model: # task is regression
                selected_model = selected_model + "_R" 
            elif classification_model and not regression_model: # task is classification
                selected_model = selected_model + "_C" #
            else: # choose from selected loss function
                if request.form.get("loss") in ["mse","mae"]:
                    selected_model = selected_model + "_R" 
                else:
                    selected_model = selected_model + "_C" 
            
            session["selected_model"] = selected_model

            # preprocess the data
            df_X,df_y = preprocess_for_model(selected_model,df_X,df_y)

            # check soem errors if the task is classification
            if model_type(selected_model) == "classification": # task is a classification task
                one_class_columns = [col for col in df_y.columns if df_y[col].nunique() < 2]
    
                if len(one_class_columns) != 0: # if there is only one value for a discrete variable that will be predicted, flash a warning
                    flash("Some of the data we are trying to predict has only one class! Please remove such classes to continue: {}".format(one_class_columns))
                    return render_template("selectAlgo_DL.html",regression_model = regression_model,
                                           classification_model = classification_model)
                if len(session.get("selected_y")) != 1:
                    flash("For classification task, only choose one variable for predicting")
                    return render_template("selectAlgo_DL.html",regression_model = regression_model,
                                        classification_model = classification_model)
                no_of_unique = len(np.unique(df_y.values))
                if no_of_unique > 2 and request.form.get("loss") == "binary_crossentropy":
                    flash("Binary crossentropy is only suitible for columns with two unique value. Selected column has {}.".format(no_of_unique))
                    return render_template("selectAlgo_DL.html",regression_model = regression_model,
                    classification_model = classification_model)
            
            # required parametrs for layer constrructing
            ALL_KEYS = ["layer-selection","activation[]","strides[]","sizes[]","units[]","ratio[]","momentum[]","epsilon[]"]
            layer_list = request.form.getlist(ALL_KEYS[0])
            activation_list = request.form.getlist(ALL_KEYS[1])
            stride_list = request.form.getlist(ALL_KEYS[2])
            size_list = request.form.getlist(ALL_KEYS[3])
            units_list = request.form.getlist(ALL_KEYS[4])
            ratio_list = request.form.getlist(ALL_KEYS[5])
            momentum_list = request.form.getlist(ALL_KEYS[6])
            epsilon_list = request.form.getlist(ALL_KEYS[7])

            parameters_list = []
            # get the data from the forms for constructing layer
            for index in range(len(layer_list)): 
                units_int = assign_default_value(units_list[index],32,"int")
                layer_stride = assign_default_value(stride_list[index],1,"int")
                size = assign_default_value(size_list[index],3,"int")
                momentum = assign_default_value(momentum_list[index],0.99,"float")
                epsilon = assign_default_value(epsilon_list[index],0.001,"float")
                ratio = assign_default_value(ratio_list[index],0.25,"float")

                parameter_list = {"layer_name" : layer_list[index], "units" : units_int, "activation" : activation_list[index],
                 "momentum" : momentum, "epsilon" : epsilon, "ratio" : ratio, "strides" : layer_stride, "size" : size}
                parameters_list += [parameter_list]
                print(parameter_list)
            
            # for image input, check if dimensions are right
            if selected_model in ["CNN_R","CNN_C"]: # determine the dimensions
                square_assumption = int(np.sqrt(len(session.get("selected_x")))) # default value is sqrt of selected_x
                width = assign_default_value(request.form.get("width"),square_assumption,check_type = "int") 
                height = assign_default_value(request.form.get("height"),square_assumption,check_type = "int")
                channel = assign_default_value(request.form.get("channel"),1,check_type = "int")
                print(square_assumption,width,height,channel)
                if height*width*channel != len(session.get("selected_x")): # dimensions does not match
                    flash("Dimension error! Please correctly input the dimensions of the image. (Dimension is {}, but {} assigned)".format(len(session.get("selected_x")),width*height*channel))
                    return render_template("selectAlgo_DL.html",regression_model = regression_model,
                    classification_model = classification_model)
                input_shape = (height,width,channel)
                session["input_shape"] = input_shape

            # create the model
            model_configurations = {"optimizer" : request.form.get("optimizer") , "loss" : request.form.get("loss") , "metrics" : ["mse"]}
            model = create_DNN(parameters_list, model_configurations, no_of_unique = no_of_unique)

            # input should be reshaped if user selected CNN as their model
            if selected_model in ["CNN_R","CNN_C"]:
                model.build((None,height,width,channel)) 

            model.summary()

            # train the model
            K = assign_default_value(request.form.get("K_cross_val"), 5, check_type = "int")
            epochs =  assign_default_value(request.form.get("epochs"), 30, check_type= "int")
            batch_size = assign_default_value(request.form.get("batch_size"), 64, check_type = "int")
            model_scores,models,(test_y_rescaled,pred_y_rescaled) = cross_validate_models(df_X,df_y,model,K,
            epochs = epochs, batch_size = batch_size, callbacks = [], optimizer = model_configurations["optimizer"], 
            loss = model_configurations["loss"], metrics = model_configurations["metrics"], input_shape = input_shape)

            if len(models) == 0:
                flash("An error occured while constructing the model!")
                return(redirect(url_for("selectAlgo_DL")))

            best_model = models[-1] #TODO : Choose best model
            session["selected_x_trained"] = session.get('selected_x')
            session["selected_y_trained"] = session.get('selected_y')
            session["model_scores"] = model_scores  # a small price 
            session["k_cross_val"] = K

            save_temp_dataframe(pd.concat([test_y_rescaled,pred_y_rescaled],axis = 1),session.get("user_id"), body = "-test-dataframe")
            save_user_model(best_model,session.get("user_id"),body = "-user-model", method = "model")

                        
            description = "[ " + str(datetime.datetime.now()) + " ] >>" + " Created new model (Model : {}). ".format(session.get("selected_model"))
            session["user_log"] += [description + ""] 

            return redirect(url_for("result"))

        return render_template("selectAlgo_DL.html", regression_model = regression_model, classification_model = classification_model)
    
    @app.route('/result')
    def result():
        """
        Show the result of the model that is trained with the test set. Metrics are shown according to the type of the models' task,
        whether it is a classification or a regression task.
        """
        
        if session.get("selected_model") == None:
            flash("No model has been trained!")
            return redirect(url_for("selectAlgo"))

        # load required models/dataframes
        result_dataframe = load_temp_dataframe(session.get('user_id'),body = "-test-dataframe")
        type_of_model = model_type(session.get("selected_model"))
        model_scores,mse_errors,mae_errors,f1_scores,log_errors = [],[],[],[],[]

        for score in session.get('model_scores'): # update list of scores for displaying
            model_scores += [score["model_scores"]] if score["model_scores"] != [] else [] # this is necessary and prevents empty rows
            mse_errors += [score["mse_errors"]] if score["mse_errors"] != [] else []
            mae_errors += [score["mae_errors"]] if score["mae_errors"] != [] else []
            f1_scores += [score["f1_scores"]] if score["f1_scores"] != [] else []
            log_errors += [score["log_errors"]] if score["log_errors"] != [] else []



        print(model_scores,mse_errors,mae_errors,f1_scores,log_errors)
        save_temp_dataframe(result_dataframe,session.get('user_id'),body = "-result-dataframe", method = "csv")
        path = os.path.join("temp",str(session.get('user_id')) + '-result-dataframe' + ".csv")

        if type_of_model == "classification": # if our task is classification
            return render_template("result.html",model_scores = model_scores,mse_errors = mse_errors,mae_errors = mae_errors,
            f1_scores = f1_scores, log_errors = log_errors,
            column_names=result_dataframe.columns.values, row_data=list(result_dataframe.head(10).values.tolist()),
            link_column="Patient ID", zip=zip, rowS = result_dataframe.shape[0], colS = result_dataframe.shape[1],
            path = path,parameters = session.get("selected_y_trained"), k_groups = range(session.get("k_cross_val")))
            
        else: # if our task is regression
            return render_template("result.html",model_scores = model_scores,mse_errors = mse_errors,mae_errors = mae_errors,
            f1_scores = f1_scores, log_errors = log_errors,
            column_names=result_dataframe.columns.values, row_data=list(result_dataframe.head(10).values.tolist()),
            link_column="Patient ID", zip=zip, rowS = result_dataframe.shape[0], colS = result_dataframe.shape[1],
            path = path,parameters = session.get("selected_y_trained"), k_groups = range(session.get("k_cross_val")))

    @app.route('/result/uploaded', methods = ["GET","POST"])
    def result_uploaded():
        """
        Upload a new file to use lastly trained model.
        """
        if session.get("selected_model") == None:
            flash("No model has been trained!")
            return redirect(url_for("selectAlgo"))

        session['referrer_page'] = 'result_uploaded'
        test_dataframe = pd.DataFrame([],columns = session.get("selected_x_trained")) # dummy data
        type_of_model = model_type(session.get("selected_model"))
        model_scores,mse_errors,mae_errors,log_errors,f1_scores = [],[],[],[],[]
        if session.get("selected_model") in ["DNN_R","DNN_C","CNN_C","CNN_R"]:
            method = "model"
        else:
            method = "pickle"
        model = load_user_model(session.get('user_id'),body = "-user-model", method = method)
        if request.method == "POST":
            df,unknown_types,unknown_columns = load_(app.config['UPLOAD_FOLDER'],request.files) # load a local file uploaded from the user
            if len(unknown_types) != 0 or len(unknown_columns) != 0: # unknown types and colums exist
                return redirect(url_for("assign_unknown_types"))

            # TODO : If y_feature exist in the data, dont drop it and calculate metrics
            file_contains_y = pd.Series(session.get('selected_y_trained')).isin(df.columns).all()
            
            if df is not None: # data loaded succesfully
                if check_suitable(test_dataframe,df): # data that trained the model and the data that is uploaded should be suitable.
                    df_X = df[session.get('selected_x_trained')]
                    if file_contains_y:
                        df_y = df[session.get('selected_y_trained')]
                    else:
                        df_y = None

                    df_X, df_y = apply_model_transformers(df_X,df_y) # apply transformers that is used in the preprocessing section for the training data
                    
                    if session.get('selected_model') in ["CNN_C","CNN_R"]: # model is CNN, input must be reshaped
                        input_shape = session.get('input_shape')
                        df_X_reshaped = df_X.values.reshape((df_X.shape[0],input_shape[0],input_shape[1],input_shape[2]))
                        y_predict = model.predict(df_X_reshaped)
                    else:
                        y_predict = model.predict(df_X)

                    if session.get('selected_model') in ["DNN_C","CNN_C"]: #classification outputs proba
                        y_predict = np.argmax(y_predict,axis=1)
                    
                    if file_contains_y is False: 
                        df_y = pd.DataFrame(y_predict,columns = session.get('selected_y_trained'))
                    else:
                        df_y = pd.DataFrame(df_y,columns = session.get('selected_y_trained'))
                        y_predict = pd.DataFrame(y_predict,columns = session.get('selected_y_trained'))

                    if type_of_model == "classification" and file_contains_y:
                        model_scores,mse_errors,mae_errors,log_errors,f1_scores = get_metrics(type_of_model,df_y,y_predict)

                    df_X,df_y = revert_model_transformers(df_X,df_y) # revert data back to display the values correctly

                    if file_contains_y:
                        _,y_predict = revert_model_transformers(y = y_predict)

                    if type_of_model == "regression" and file_contains_y:
                        model_scores,mse_errors,mae_errors,log_errors,f1_scores = get_metrics(type_of_model,df_y,y_predict)

                    for col in df_y.columns: 
                        if file_contains_y:
                            df_X[col + "_original"] = df_y[col]
                            df_X[col + "_pred"] = y_predict[col]
                        else:
                            df_X[col] = df_y[col]

                    result_dataframe = df_X

                    # save the dataframe and display the results
                    save_temp_dataframe(result_dataframe,session.get('user_id'),body = "-result-dataframe", method = "csv")
                    path = os.path.join("temp",str(session.get('user_id')) + '-result-dataframe' + ".csv")
                    
                    return render_template("result_user.html",
                    column_names=result_dataframe.columns.values, row_data=list(result_dataframe.head(10).values.tolist()),
                    link_column="Patient ID", zip=zip, rowS = result_dataframe.shape[0], colS =result_dataframe.shape[1],
                    path = path, parameters = session.get("selected_y_trained"), dataSelected = True,model_scores = model_scores,
                    f1_scores = f1_scores, mse_errors = mse_errors, mae_errors = mae_errors, log_errors = log_errors, file_contains_y = file_contains_y)
                else: 
                    flash("Data that is used is not suitable with model! Please check the parameters.")
                    return render_template("result_user.html", dataSelected = False)
        return render_template("result_user.html")
        

    @app.route('/scatter_graph', methods = ["GET","POST"])
    def scatter_graph():
        """
        Create scatter graph with selected parameters. Up to 10 parameters can be selected.
        """
        df = load_temp_dataframe(session.get("user_id"))
        if df.shape[0] == 0: # current data is empty
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
            if len(selected_features) >= 10: # more than 10 parameter is selected
                flash("More than 10 columns has been selected. Please select number of features between 2 and 10")
                return render_template('graphs/scatter_plot.html',columns = df.columns)

            elif len(selected_features) <= 2: # less than 2 parameter is selected
                flash("Less than 10 colums has been selected. Please select number of features between 2 and 10")
                return render_template('graphs/scatter_plot.html',columns = df.columns)
            
            else: # everything is fine, return graph
                description = "[ " + str(datetime.datetime.now()) + " ] >>" + " Scatter matrix is created with   " + str(len(selected_features)) +" variables."
                session["user_log"] += [description + ""]
                return scatter_matrix(df,selected_features)
        return render_template('graphs/scatter_plot.html',columns = df.columns)


    @app.route('/correlation_graph', methods = ["GET","POST"])
    def correlation_graph():
        """
        Create correlation graph between selected parameters.
        """
        df = load_temp_dataframe(session.get("user_id"))
        if df.shape[0] == 0:
            flash("No data is selected. Please upload or select data to continue.")
            return redirect(url_for("workspace"))

        if request.method == "POST":
            if 'parameters' in request.form:
                selected_features = request.form.getlist('parameters')
                description = "[ " + str(datetime.datetime.now()) + " ] >>" + " Correlation matrix is created with   " + str(len(selected_features)) +" variables."
                session["user_log"] += [description + ""]
                return correlation_plot(df.select_dtypes(exclude = ['object']),selected_features) # objects are excluded since they are discrete
                
            else:
                flash("Please select parameters to create a correlation plot.")
                return redirect(url_for("correlation_graph"))

        return render_template('graphs/correlation_plot.html',columns =df.select_dtypes(exclude = ['object']).columns)


    @app.route('/pie_graph', methods = ["GET","POST"])
    def pie_graph():
        """
        Create pie graph for selected parameters. Maximum value that can be displayed is 255.
        """
        df = load_temp_dataframe(session.get("user_id"))
        if df.shape[0] == 0:
            flash("No data is selected. Please upload or select data to continue.")
            return redirect(url_for("workspace"))

        if request.method == "POST":
            if request.form.getlist("parameters") != []:
                if request.form.get("sort-values"): # should pie graph be sorted?
                    sort_values = True
                else:
                    sort_values = False               
                top_values =  255 if check_float(request.form.get('head_number')) == False else int(request.form.get('head_number')) # num of sections to be displayed
                description = "[ " + str(datetime.datetime.now()) + " ] >>" + " Pie graph is created with   " + str(len(request.form.getlist('parameters'))) +" variables."
                session["user_log"] += [description + ""]
                return pie_plot(df.select_dtypes(include = ["object","int32","int64"]),request.form.getlist("parameters"),sort_values,top_values)

            else:
                flash("Please select parameters to process.")
                return redirect(url_for('pie_graph'))

        return render_template('graphs/pie_plot.html',columns = df.select_dtypes(include = ["object","int32","int64"]).columns)

    @app.route('/dist_graph', methods = ["GET","POST"])
    def dist_graph():
        """
        Create a histogram. A number of bin can be inputted by users to control the graph.
        """
        df = load_temp_dataframe(session.get("user_id"))
        if df.shape[0] == 0:
            flash("No data is selected. Please upload or select data to continue.")
            return redirect(url_for("workspace"))
        if not session.get('user_log'):
            session['user_log'] = []
            
        if request.method == "POST":
            if 'selected_parameter' in request.form:
                numberBin = 20 if (request.form['numberBin'].isnumeric() == False) else int(request.form['numberBin']) # number of bin
                description = "[ " + str(datetime.datetime.now()) + " ] >>" + " Distribution graph is created for   " + request.form['selected_parameter']
                session["user_log"] += [description + ""]
                return dist_plot(df.select_dtypes(exclude = ['object']),request.form['selected_parameter'],numberBin)
            else:
                flash("Please select parameters to create distribution graph.")
                return redirect(url_for("dist_graph"))
        return render_template('graphs/dist_plot.html',columns = np.sort(df.select_dtypes(exclude = ['object']).columns))

    @app.route('/bar_graph', methods = ["GET","POST"])
    def bar_graph():
        """
        Create a bar graph. Two options can be selected by user - vertical and horizontal.
        """
        df = load_temp_dataframe(session.get("user_id"))
        if df.shape[0] == 0:
            flash("No data is selected. Please upload or select data to continue.")
            return redirect(url_for("workspace"))
        if not session.get('user_log'):
            session['user_log'] = []
            
        if request.method == "POST":
            if request.form.getlist("parameters") != []:
                if 'selected_type' in request.form: # horizontal or vertical graph?
                    selectedType = request.form["selected_type"]
                else:
                    selectedType = "Horizontal"
                description = "[ " + str(datetime.datetime.now()) + " ] >>" + " Bar graph is created with   " + str(len(request.form.getlist("parameters"))) +" variables."
                session["user_log"] += [description + ""]
                return bar_plot(df.select_dtypes(include = ['object']),request.form.getlist("parameters"),selectedType)
            else:
                flash("Please select parameters to create bar plot.")
                return redirect(url_for("bar_graph"))
        return render_template('graphs/bar_plot.html',columns = np.sort(df.select_dtypes(include = ['object']).columns))


    @app.route("/pca_transform", methods = ["GET","POST"])
    def pca_transform():
        """
        Apply PCA transform to the data.
        """
        df = load_temp_dataframe(session.get("user_id"))
        if df.shape[0] == 0:
            flash("No data is selected. Please upload or select data to continue.")
            return redirect(url_for("workspace"))
        if not session.get('user_log'):
            session['user_log'] = []
            
        if request.method == "POST":
            if request.form["n_component"].isnumeric(): # number of component -- final dimension that is reduced
                n_of_component = int(request.form["n_component"]) 
                if int(n_of_component) > len(df):
                    n_of_component = len(df)
            
                elif int(n_of_component) <= 0:
                    flash("Invalid number of component! Enter a positive number.")
                    return redirect("pca_transform")

            else:
                n_of_component = None

            
            if request.form["variance_ratio"] != "": # check if variance ratio is given -- this will automatically determine the final dimension. 
                variance_ratio = float(request.form["variance_ratio"])
            else:
                variance_ratio = None

            if df.isnull().sum().sum() != 0: # if there are NaN values
                flash("NaN value detected!")
                return redirect("pca_transform")


            df,pca = PCA_transformation(data = df, reduce_to = n_of_component, var_ratio = variance_ratio)
            save_temp_dataframe(df,session.get("user_id"))
            session["selected_x"] = []
            session["selected_y"] = []
            #This user-log system will be changed later on.
            description = "[ " + str(datetime.datetime.now()) + " ] >>" + " PCA transformation is applied (Ratio = {}, Number of component = {} ).".format(variance_ratio,n_of_component)
            session["user_log"] += [description + ""]
            return PCA_transformation_describe(df,pca)

        return render_template("transformation/pca_transform.html")


    @app.route("/create_column", methods = ["GET","POST"])
    def create_column():
        """
        Create, update, change the columns with existing columns of the currently used DataFrame.
        """
        df = load_temp_dataframe(session.get("user_id"))
        if df.shape[0] == 0:
            flash("No data is selected. Please upload or select data to continue.")
            return redirect(url_for("workspace"))

        if not session.get('user_log'):
            session['user_log'] = []
        if request.method == "POST":
            selected_parameters = request.form.getlist("selected_parameters")
            selected_mode = request.form["selected_mode"] # operation that will be done on selected parameters
            if request.form.get("delete_columns"): # should we delete used columns?
                delete_columns = True
            else:
                delete_columns = False
            new_column_name = request.form.get("new_column_name")
            
            if new_column_name in df.columns and new_column_name is not "" : # -- check if column name already exist
                flash("This column name is already exist!")
                return redirect("create_column")

            elif (new_column_name == None or new_column_name == "") and (selected_mode in ["sum","mean","difference","concat"]): # -- check if column name is not entered
                flash("Please enter a column name!")
                return redirect("create_column")

            if selected_parameters == []: # -- no parameter is given
                flash("Please select parameters!")
                return redirect("create_column")
            
            if selected_mode == "Impute":
                selected_mode = request.form.get('selected_mode_impute')

            print(selected_mode)
            df,transformer = combine_columns(data = df, selected_columns = selected_parameters,
            new_column_name = new_column_name, mode = selected_mode, delete_column = delete_columns) 
            save_temp_dataframe(df,session.get("user_id"))
            session["selected_x"] = []
            session["selected_y"] = []
            description = "[ " + str(datetime.datetime.now()) + " ] >>" + " New column is created. ( Mode ={}, New colum name = {} )".format(selected_mode,new_column_name)
            session["user_log"] += [description + ""]

        return render_template("transformation/create_column.html", columns = df.columns)

    @app.route("/filter_transform", methods = ["GET","POST"])
    def filter_transform():
        """
        Filter the DataFrame. This function selects particular rows from DataFrame by applying queries.
        For object columns, values can be selected.
        For continious columns, boundaries can be set.
        """
        df = load_temp_dataframe(session.get("user_id"))
        first_shape = df.shape
        if df.shape[0] == 0:
            flash("No data is selected. Please upload or select data to continue.")
            return redirect(url_for("workspace"))
        if not session.get('user_log'):
            session['user_log'] = []
            
        if request.method == "POST":
            actions = {}
            for col in df.columns: # remove columns that will not affect the filtered data
                col_list = remove_empty_lists(request.form.getlist(col))
                if col_list:
                    actions[col] = col_list
                    
            df = filter_data(df,actions) # main function, filters the data and returns the changed version
            after_shape = df.shape
            save_temp_dataframe(df,session.get("user_id"))
            session["selected_x"] = [] # reset selected parameters 
            session["selected_y"] = []
            description = "[ " + str(datetime.datetime.now()) + " ] >>" + " Parameters are filtered. Number of rows in data is changed from {} to {}.".format(first_shape[0],after_shape[0]) 
            session["user_log"] += [description + ""]
        return render_template("transformation/filter_transform.html", cols = df.columns, objectCols = df.select_dtypes(include = "object").columns, df = df)

    @app.route('/download_csv')
    def download_csv():
        """
        Download a csv file given with argument :path: This function is only used in other functions.
        """
        path = request.args.get('path')
        if path is None:
            flash("An error occured during downloading file!")
            return redirect(url_for('workspace'))
        return send_file(path,mimetype="text/csv",attachment_filename='myfile.csv',as_attachment=True)


    app.register_blueprint(bp, url_prefix='/auth')
    init_app(app)
    return app



