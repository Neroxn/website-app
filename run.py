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
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_squared_log_error 
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
    
        df = load_temp_dataframe(session.get("user_id")) #load dataframe

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
                save_temp_dataframe(df,session.get("user_id"))
                session["selected_workspace"] = None
                session["selected_dataframe"] = None
                return redirect(url_for("workspace"))

            elif request.form.get('Clear Log'):
                session['user_log'] = []
                return redirect(url_for("workspace"))

            elif request.form.get('Download DataFrame'):
                path = str(session.get("user_id")) + "_" + str(session.get("selected_workspace")) + "_" + str(session.get("selected_dataframe")) + ".csv"
                path = "temp/" + path 
                return redirect(url_for("download_csv",path=path))
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
        df = load_temp_dataframe(session.get("user_id"))
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
                save_temp_dataframe(df,session.get("user_id"))
                create_workspace(session["user_id"], df)
                

                #Redirect to result page back if it comes from result page
                if request.args.get("result"):
                    model = load_user_model(session["user_id"])

                    if session["selected_algo"] == "SVM":
                        df2 = df[session["selected_x"]+session["selected_y"]]
                        df2 = dropNanAndDuplicates(df2, 0.75)       
                        actual = df2[session["selected_y"]]
                        df2, encoderArr = stringEncoder(df2, df2.loc[:, df2.dtypes == object].columns)
                        df2.values = session["scaler"].transform(df2.values)
                        testX = df2[session["selected_x"]]
                        result = model.predict(testX).reshape((len(testX),-1))
                        result = session["scalerY"].inverse_transform(result)
                        if encoderArr:
                            result = stringDecoder(result, encoderArr, session["selected_y"])
                    elif session["selected_algo"] == "RandomForest":
                        df2 = df[session["selected_x"]+session["selected_y"]]
                        df2 = dropNanAndDuplicates(df2, 0.75)
                        actual = df2[session["selected_y"]]
                        df2, encoderArr = stringEncoder(df2, df2.loc[:, df2.dtypes == object].columns)
                        testX = df2[session["selected_x"]]
                        result = model.predict(testX).reshape((len(testX),-1))
                        if encoderArr:
                            result = stringDecoder(result, encoderArr, session["selected_y"])
                    elif session["selected_algo"] == "Adaboost":
                        df2 = df[session["selected_x"]+session["selected_y"]]
                        df2 = dropNanAndDuplicates(df2, 0.75)
                        actual = df2[session["selected_y"]]
                        df2, encoderArr = stringEncoder(df2, df2.loc[:, df2.dtypes == object].columns)
                        testX = df2[session["selected_x"]]
                        result = model.predict(testX).reshape((len(testX),-1))
                        if encoderArr:
                            result = stringDecoder(result, encoderArr, session["selected_y"])
                            
                    save_temp_dataframe(actual,session["user_id"],body="-actual-y")
                    save_temp_dataframe(result,session["user_id"],body="-result-y")
                    return redirect(url_for("result"))

                description = str(datetime.datetime.now()) + " created new workspace. "
                session["user_log"] += [description + user_log_information(session)] 
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
            session["user_log"] += [description + user_log_information(session)]
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

        if request.method == 'POST':
            session["selected_y"] = request.form.getlist('hello')
            description = str(datetime.datetime.now()) + " Selected  " + str(len(session['selected_y'])) + " many target for the model."
            session["user_log"] += [description + user_log_information(session)]
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
        if not session.get("selected_x"):
            session["selected_x"] = []

        if not session.get("selected_y"):
            session["selected_y"] = []

        no_of_integer,no_of_inexact,no_of_object,other_columns = instance_divider(df[session.get("selected_y")])
        if other_columns != 0:
            flash("A variable y with no possible model selection has found!")
            return redirect(url_for("select_y"))
        else:
            if no_of_integer != 0: 
                if no_of_inexact == 0  and no_of_object != 0: #All columns are integer or object, use classification
                    classification_model = True
                    regression_model = False

                elif no_of_object == 0: # All columns are integer or float, use regression
                    regression_model = True 
                    classification_model = False
                
                elif no_of_inexact != 0 and no_of_object != 0: #Columns are mixed, show error
                    flash("No possible model can be selected! Please use different target variables for prediction")

                else: #All columns are integer. Both of regression and classification can be used
                    regression_model = True
                    classification_model = True
            else: 
                if no_of_inexact == 0  and no_of_object != 0: #All columns are object, use classification
                    classification_model = True
                    regression_model = False

                elif no_of_object == 0: # All columns are flaot, use regression
                    regression_model = True 
                    classification_model = False

      
        if request.method == "POST":
            # get the df_X and df_y
            df_X = df[session.get("selected_x")]
            df_y = df[session.get("selected_y")]

            selected_model = request.form.get("selected_model")

            ##PART 1##
            print(selected_model,"Selected-model successfull")
            selected_parameters = request.form
            print(selected_parameters)
            df_X,df_y = preprocess_for_model(selected_model,df_X,df_y) # apply preprocess that is needed
            print("Preprocess-model successfull")
            ##PART 2##
            model = fetch_model(selected_model,selected_parameters) # create model with given parameters
            print("Fetch-model successfull")

            ##PART 3##
            train_X, test_X, train_y, test_y = train_test_split(df_X,df_y,train_size = 0.80) # split the dataframe
            model = train_model(model,train_X,train_y) # train the model 
            print("Train-model successfull")

            ##PART 4##
            predicted_y = test_model(model,test_X) # predict the test_x
            print("Test-model successfull")

            ##PART 5##
            # prepare result dataframe to show our performance
            test_y.columns  = [col + "_actual" for col in test_y.columns]
        
            predicted_y = pd.DataFrame(predicted_y,columns = train_y.columns)
            predicted_y.columns = [col + "_predicted" for col in test_y.columns]
            predicted_y.set_index([test_y.index],inplace=True)   
            result_dataframe = pd.concat([test_y,predicted_y],axis=1)
            print(predicted_y.head())
            print(test_y.head())
            print(result_dataframe.head())
            # save model and dataframe
            save_user_model(model,session.get("user_id"),body = "test-model")

            save_temp_dataframe(result_dataframe,session.get("user_id"), body = "-result-dataframe")
            return "Succesfull"

        return render_template("select_algo.html",regression_model = regression_model, 
        classification_model = classification_model,more_than_one = len(session.get('selected_y')) > 1)


        
    @app.route('/scatter_graph', methods = ["GET","POST"])
    def scatter_graph():
        df = load_temp_dataframe(session.get("user_id"))

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
                description = str(datetime.datetime.now()) + " Scatter matrix is created with   " + str(len(selected_features)) +" variables."
                session["user_log"] += [description + user_log_information(session)]
                return scatter_matrix(df,selected_features)
        return render_template('graphs/scatter_plot.html',columns = df.columns)


    @app.route('/correlation_graph', methods = ["GET","POST"])
    def correlation_graph():
        df = load_temp_dataframe(session.get("user_id"))
        if request.method == "POST":
            if 'parameters' in request.form:
                selected_features = request.form.getlist('parameters')
                description = str(datetime.datetime.now()) + " Correlation matrix is created with   " + str(len(selected_features)) +" variables."
                session["user_log"] += [description + user_log_information(session)]
                return correlation_plot(df.select_dtypes(exclude = ['object']),selected_features)
                
            else:
                return render_template('graphs/correlation_plot.html',columns = df.select_dtypes(exclude = ['object']).columns,error = "Please select parameters to process.")

        return render_template('graphs/correlation_plot.html',columns =df.select_dtypes(exclude = ['object']).columns)


    @app.route('/pie_graph', methods = ["GET","POST"])
    def pie_graph():
        df = load_temp_dataframe(session.get("user_id"))
        if request.method == "POST":
            if request.form.getlist("parameters") != []:
                if request.form.get("sort-values"):
                    sort_values = True
                else:
                    sort_values = False

                description = str(datetime.datetime.now()) + " Pie graph is created with   " + str(len(request.form.getlist('parameters'))) +" variables."
                session["user_log"] += [description + user_log_information(session)]
                return pie_plot(df.select_dtypes(include = ["object"]),request.form.getlist("parameters"),sort_values)

            else:
                flash("Please select parameters to process.")
                return redirect('pie_plot')

        return render_template('graphs/pie_plot.html',columns = df.select_dtypes(include = ["object"]).columns)

    @app.route('/dist_graph', methods = ["GET","POST"])
    def dist_graph():
        df = load_temp_dataframe(session.get("user_id"))

        if not session.get('user_log'):
            session['user_log'] = []
        if request.method == "POST":
            print(request.form)
            if 'selected_parameter' in request.form:
                numberBin = 20 if (request.form['numberBin'].isnumeric() == False) else int(request.form['numberBin'])
                description = str(datetime.datetime.now()) + " Scatter matrix is created for   " + request.form['selected_parameter']
                session["user_log"] += [description + user_log_information(session)]
                return dist_plot(df.select_dtypes(exclude = ['object']),request.form['selected_parameter'],numberBin)
            else:
                return render_template('graphs/dist_plot.html',columns =df.select_dtypes(exclude = ['object']).columns, error = "Please choose the parameter for histogram!")
        return render_template('graphs/dist_plot.html',columns = df.select_dtypes(exclude = ['object']).columns)

    @app.route('/bar_graph', methods = ["GET","POST"])
    def bar_graph():
        df = load_temp_dataframe(session.get("user_id"))
        if not session.get('user_log'):
            session['user_log'] = []
        if request.method == "POST":
            print(request.form)
            if request.form.getlist("parameters") != []:
                if 'selected_type' in request.form:
                    selectedType = request.form["selected_type"]
                else:
                    selectedType = "Horizontal"
                description = str(datetime.datetime.now()) + " Scatter matrix is created with   " + str(len(request.form.getlist("parameters"))) +" variables."
                session["user_log"] += [description + user_log_information(session)]
                return bar_plot(df.select_dtypes(include = ['object']),request.form.getlist("parameters"),selectedType)
            else:
                return render_template('graphs/bar_plot.html',columns =df.select_dtypes(include = ['object']).columns)
        return render_template('graphs/bar_plot.html',columns = df.select_dtypes(include = ['object']).columns)


    @app.route("/pca_transform", methods = ["GET","POST"])
    def pca_transform():
        df = load_temp_dataframe(session.get("user_id"))

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
            #This user-log system will be changed later on.
            description = str(datetime.datetime.now()) + " PCA transformation is applied."
            session["user_log"] += [description + user_log_information(session)]
            return PCA_transformation_describe(df,pca)

        return render_template("transformation/pca_transform.html")


    @app.route("/create_column", methods = ["GET","POST"])
    def create_column():
        df = load_temp_dataframe(session.get("user_id"))

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
            description = str(datetime.datetime.now()) + " New column is created."
            session["user_log"] += [description + user_log_information(session)]

        return render_template("transformation/create_column.html", columns = df.columns)

    @app.route("/filter_transform", methods = ["GET","POST"])
    def filter_transform():
        df = load_temp_dataframe(session.get("user_id"))
        if not session.get('user_log'):
            session['user_log'] = []
        if request.method == "POST":
            actions = {}
            for col in df.columns:
                    actions[col] = request.form.getlist(col)
            df = filter_data(df,actions).to_dict('list')
            save_temp_dataframe(df,session.get("user_id"))
            description = str(datetime.datetime.now()) + " Parameters are filtered." 
            session["user_log"] += [description + user_log_information(session)]
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
        if request.method == "POST":
            #Get test data
            if request.form.get("Upload test data"):
                return redirect(url_for("upload_file", result = True))
                
        #Get actual and prediction Y
        actual = load_temp_dataframe(session["user_id"],body="-actual-y")
        pred = load_temp_dataframe(session["user_id"],body="-result-y")
        
        print(pred.head(5))
        print(actual.head(5))
        print(actual.shape,pred.shape)
        #Check whether actual and pred are None
        if actual is None or pred is None:
            flash("Please select variables again")
            return redirect(url_for("select_variables"))
        
        mae = []
        mse = []
        rmse = []
        msle = []
        mape = []
        #Label columns as actual and prediction / calculate error metrics
        for col in actual.columns:
            mae.append("Mean absolute error of " + col + " is: "+ str(mean_absolute_error(actual[col], pred[col])))
            mse.append("Mean square error of " + col + " is: "+ str(mean_squared_error(actual[col], pred[col])))
            #msle.append("Mean squared logarithmic error of " + actual.columns[i] + " is: "+ str(mean_squared_log_error(actual[actual.columns[i]], pred[pred.columns[i]])))
            
        actual.columns = ["Actual "+ col for col in actual.columns]
        pred.columns = ["Predicted "+ col for col in pred.columns]
        
        #Concat 2 df
        df = pd.concat([actual, pred], axis = 1)
        
        
        isLoaded = True
        return render_template("result.html", column_names=df.columns.values, row_data=list(df.head(5).values.tolist()), link_column="Patient ID", zip=zip, isLoaded = isLoaded, rowS = df.shape[0], colS = df.shape[1], mae = mae, mse = mse, rmse = rmse, msle = msle, mape = mape)

    @app.errorhandler(403)
    def forbidden(e):
        return render_template('error/403.html'), 403


    @app.errorhandler(404)
    def page_not_found(e):
        return render_template('error/404.html'), 404


    @app.errorhandler(500)
    def internal_server_error(e):
        return render_template('error/500.html'), 500

    app.register_blueprint(bp, url_prefix='/auth')
    init_app(app)
    return app

###########################
# TO DO LIST : 

# X> Way to handle with scalers-encoders - we should use same scalers and encoders in testing data. (5)
#   >> Note that dataframe can be a little bit different than test data since we can do transformations
#   .. in this case, we can either apply same transformations or expect that test data columns were same.
#   .. Lets assume that user did created new columns but did not apply transformations (encoders or scalers)
#   .. So we have to apply transformations in reverse order.  
#   .. For memorizing the mod operations, a new system can be created so that we would not need any assumptions 
#   .. and we can apply it to the new-uploaded dataframe without asking but this will be handled later.

# X> Scaling will be used only when its needed -- for y, not for performence. (4)
#   >> This is done mostly - we only need a way to apply pipe of transformations as mentioned above

# X> A seperete choice for every dtypes should exist. So selecting all objects-ints-floats could be better/faster (3)
#   >> This is important as we let user choose their feature for many things. 

# 
# X> Add result page. (6)



# X> Add Boostrap for better web design
############################


