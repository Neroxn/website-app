from bokeh.models import ColumnDataSource
from bokeh.resources import INLINE
from bokeh.embed import components
from bokeh.plotting import figure
from bokeh.layouts import column, gridplot
from bokeh.palettes import RdBu as colors
from bokeh.models import ColorBar, LinearColorMapper
from utils import *

def scatter_matrix(dataset,features):
    """
    Create a scatter matrix by using Bokeh library.
    Parameters:
    :dataset: --  Dataframe we are using
    :features: -- Features that will be used in plotting the graph.

    Return:
    Render a template that includes graph in return
    """

    TOOLS = "box_select,lasso_select,pan,wheel_zoom,box_zoom,reset,help,save"
    dataset_selected = dataset[features]
    dataset_source = ColumnDataSource(data=dataset)
    scatter_plots = []
    y_max = len(dataset_selected.columns)-1
    for i, y_col in enumerate(dataset_selected.columns):
        for j, x_col in enumerate(dataset_selected.columns[::-1]) :
            p = figure(plot_width=100, plot_height=100, x_axis_label=x_col, y_axis_label=y_col)
            p.circle(source=dataset_source,x=x_col, y=y_col, fill_alpha=0.3, line_alpha=0.3, size=3)
            if j > 0:
                p.yaxis.axis_label = ""
                p.yaxis.visible = False
                p.y_range = linked_y_range
            else:
                linked_y_range = p.y_range
                p.plot_width=240
            if i < y_max:
                p.xaxis.axis_label = ""
                p.xaxis.visible = False
            else:
                p.plot_height=160
            if i > 0:
                p.x_range = scatter_plots[j].x_range

            scatter_plots.append(p)
    #scatter_plots = list(np.flipud(scatter_plots))
    grid = gridplot(scatter_plots, ncols = len(dataset_selected.columns))

    script, div = components(grid)
    return render_template(
    'graphs/scatter_plot.html',
    plot_script=script,
    plot_div=div,
    js_resources=INLINE.render_js(),
    css_resources=INLINE.render_css(),
    x_name = "t1",
    y_name = "t2",
    graphSelected = True,
    columns = dataset.columns
    ).encode(encoding='UTF-8')


def correlation_plot(df,selected_parameters):
    """
    Create a correlation plot by using Bokeh library.
    Parameters:
    :dataset: --  Dataframe we are using
    :features: -- Features that will be used in plotting the graph.

    Return:
    Render a template that includes graph in return
    """
    import pandas as pd
    from bokeh.io import output_file, show
    from bokeh.models import BasicTicker, ColorBar, LinearColorMapper, ColumnDataSource, PrintfTickFormatter
    from bokeh.plotting import figure
    from bokeh.transform import transform
    from bokeh.palettes import Viridis256

    # Read your data in pandas dataframe
    data = df[selected_parameters]
    #Now we will create correlation matrix using pandas
    data_corr = data.corr()

    data_corr.index.name = 'AllColumns1'
    data_corr.columns.name = 'AllColumns2'

    # Prepare data.frame in the right format
    data_corr = data_corr.stack().rename("value").reset_index()
    print(selected_parameters)

    # I am using 'Viridis256' to map colors with value, change it with 'colors' if you need some specific colors
    mapper = LinearColorMapper(
        palette=Viridis256, low=data_corr.value.min(), high=data_corr.value.max())

    # Define a figure and tools
    TOOLS = "box_select,lasso_select,pan,wheel_zoom,box_zoom,reset,help,save"
    p = figure(
        tools=TOOLS,
        plot_width=1500,
        plot_height=1250,
        title="Correlation plot",
        x_range=list(data_corr.AllColumns1.drop_duplicates()),
        y_range=list(data_corr.AllColumns2.drop_duplicates()),
        toolbar_location="right",
        x_axis_location="below")

    # Create rectangle for heatmap
    p.rect(
        x="AllColumns1",
        y="AllColumns2",
        width=1,
        height=1,
        source=ColumnDataSource(data_corr),
        line_color=None,
        fill_color=transform('value', mapper))
        
    p.xaxis.major_label_orientation = "vertical"
    # Add legend
    color_bar = ColorBar(
        color_mapper=mapper,
        location=(0, 0),
        ticker=BasicTicker(desired_num_ticks=10))

    p.add_layout(color_bar, 'right')

    path = "temp/" + str(session.get('user_id')) + "-temp.csv"
    data_corr.to_csv(path)

    script, div = components(p)
    return render_template(
    'graphs/correlation_plot.html',
    plot_script=script,
    plot_div=div,
    js_resources=INLINE.render_js(),
    css_resources=INLINE.render_css(),
    graphSelected = True,
    columns = df.columns,
    path = path
    ).encode(encoding='UTF-8')


def create_feature_matrix(data,parameters):
    """
    Create feature matrix with objects. Example:
    If parameters = [gender,eye]
    Resulting list should be [[M,LEFT],[M,RİGHT],[W,LEFT],[W,RİGHT]]
    """
    import itertools 

    unique_list = []
    for parameter in parameters: # Collect unique parameters
        unique_list += [data[parameter].unique()]


    all_parameter_permutations = list(itertools.product(*unique_list)) # Create permutation of features
    result = []
    for permutation in all_parameter_permutations: #Find the count of all conditions
        condition = np.full(len(data),True)
        name = ""
        for i in range(len(parameters)):
            condition = (condition & (data[parameters[i]] == permutation[i]))
            name += (parameters[i] + "=" + str(permutation[i]) + ",")
        result += [[name,data.loc[condition].shape[0]]]
    return result   

def pie_plot(data,selected_parameter, sort_by_values = False, top_values = 255):
    from math import pi 
    from bokeh.transform import cumsum 
    from bokeh.palettes import inferno
    result = create_feature_matrix(data,selected_parameter)
    df_pie_agg = pd.DataFrame(result,columns = ["Parameter","Count"])
    
    if top_values > 255 or top_values < 1:
        flash("Values to be displayed is out of bound. It is set to it's default value (255).")
        top_values = 255


    # Add angles based on Win Count so each wedge is the right size
    df_pie_agg['Angle'] = df_pie_agg['Count']/df_pie_agg['Count'].sum() * 2*pi
    if top_values > df_pie_agg.shape[0]:
        top_values = df_pie_agg.shape[0]
        
    color_series = inferno(top_values)
    if sort_by_values:
        df_pie_agg = df_pie_agg.sort_values(by = 'Count', ascending = False)

    df_pie_agg = df_pie_agg.head(top_values)
    df_pie_agg['color'] = color_series
    TOOLS = "box_select,lasso_select,pan,wheel_zoom,box_zoom,reset,help,save"
    # Draw a chart
    p = figure(title='Pie Chart', x_range=(-0.5, 1.0),
            plot_width=800, plot_height=600,tools = TOOLS, 
            toolbar_location="below", tooltips="@Parameter: @{Count}") 

    p.wedge(x=0.1, y=1, radius=0.4,
            start_angle=cumsum('Angle', include_zero=True), 
            end_angle=cumsum('Angle'),
            line_color="white", 
            legend='Parameter', 
            fill_color = 'color',
            source=df_pie_agg)

    path = "temp/" + str(session.get('user_id')) + "-temp.csv"

    df_pie_agg.to_csv(path)
    script, div = components(p)
    return render_template(
    'graphs/pie_plot.html',
    plot_script=script,
    plot_div=div,
    js_resources=INLINE.render_js(),
    css_resources=INLINE.render_css(),
    graphSelected = True,
    selected = selected_parameter,
    columns = data.columns,
    path = path
    ).encode(encoding='UTF-8')

def dist_plot(df,parameter,bins = 20):
    """
    Create a distribution plot by using Bokeh library.
    Parameters:
    :df: --  Dataframe we are using
    :parameter: -- Features that will be used in plotting the graph.
    :bins: -- Total number of bins that will be used

    Return:
    Render a template that includes graph in return
    """
    # Use numpy to create histogram bins
    hist, edges = np.histogram(df[parameter], bins=bins)
    TOOLS = "box_select,lasso_select,pan,wheel_zoom,box_zoom,reset,help,save"
    # Draw a chart
    p = figure(title='Histogram', plot_width=800, plot_height=600,
            x_axis_label=parameter, y_axis_label='Count', tools = TOOLS)

    p.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:], line_color='white', fill_color='black')


    path = "temp/" + str(session.get('user_id')) + "temp.csv"
    pd.DataFrame(np.c_[hist,edges[:-1]],columns = ["hist","edges"]).to_csv(path)

    script, div = components(p)
    return render_template(
    'graphs/dist_plot.html',
    plot_script=script,
    plot_div=div,
    js_resources=INLINE.render_js(),
    css_resources=INLINE.render_css(),
    graphSelected = True,
    columns = df.columns,
    selected = parameter,
    path=path
    ).encode(encoding='UTF-8')

def nan_plot(df):
    return "Show Nan"

def confusion_matrix_plot(y_trues,y_preds):
    """
    Confusion matrix plot used in result page.
    """
    from bokeh.io import output_file, show
    from bokeh.models import BasicTicker, ColorBar, LinearColorMapper, ColumnDataSource, PrintfTickFormatter
    from bokeh.plotting import figure
    from bokeh.transform import transform
    from sklearn.metrics import confusion_matrix
    figures = []
    for y_true_col,y_pred_col in zip(y_trues.columns,y_preds.columns):
        y_true = y_trues[y_true_col]
        y_pred = y_preds[y_pred_col]
        decoded_labels = sorted(list(set(y_true)))
        #print("Decoded labels are : ", decoded_labels)
        #print(confusion_matrix(y_true,y_pred,labels = decoded_labels))

        df = pd.DataFrame(confusion_matrix(y_true,y_pred),columns = ["Predicted " + str(col) for col in decoded_labels], index = ["True " + str(col) for col in decoded_labels])
        df.index.name = "Labels"
        df.columns.name = "Predictions"
        
        df = df.stack().rename("value").reset_index()
        #print(df)
        
        # Had a specific mapper to map color with value
        mapper = LinearColorMapper(
            palette='Viridis256', low=df.value.min(), high=df.value.max())
        # Define a figure
        p = figure(
            plot_width=int(1000/len(y_trues.columns)),
            plot_height=int(600/len(y_trues.columns)),
            title="Heatmap of the parameter :" + y_true_col,
            x_range=sorted(list(df.Predictions.drop_duplicates())),
            y_range=(sorted(list(df.Labels.drop_duplicates()))[::-1]),
            tools="box_select,lasso_select,pan,wheel_zoom,box_zoom,reset,help,save",
            x_axis_location="above",
            toolbar_location="below", tooltips="@{value}")
        # Create rectangle for heatmap
        p.rect(
            x="Predictions",
            y="Labels",
            width=1,
            height=1,
            source=ColumnDataSource(df),
            line_color=None,
            fill_color=transform('value', mapper))
        # Add legend
        color_bar = ColorBar(
            color_mapper=mapper,
            location=(0, 0),
            ticker=BasicTicker(desired_num_ticks=len(colors)))
        p.add_layout(color_bar, 'right')
        figures += [p]
    print(figures)
    grid = gridplot(figures,ncols = int(np.ceil(len(figures)**(0.5))))
    return components(grid)

def bar_plot(data,selected_parameter, option = 'Vertical'):
    """
    Create a bar plot by using Bokeh library.
    Parameters:
    :data: --  Dataframe we are using
    :selected_parameter: -- Features that will be used in plotting the graph.
    :option: -- Vertical or Horizontal.

    Return:
    Render a template that includes graph in return
    """
    # Draw a chart
    TOOLS = "box_select,lasso_select,pan,wheel_zoom,box_zoom,reset,help,save"
    result = create_feature_matrix(data,selected_parameter)
    df_pie_agg = pd.DataFrame(result,columns = ["Parameter","Count"])
    print(df_pie_agg['Parameter'])

    if option == 'Vertical':
        p = figure(title='Vertical Bar Chart', x_range=df_pie_agg["Parameter"], 
                plot_width=900, plot_height=600,
                y_axis_label='Count',tooltips="@Parameter: @{Count}",tools = TOOLS)

        p.vbar(x="Parameter", width = 0.75, bottom=0, top= "Count", 
            fill_color='black', line_color='white',source = df_pie_agg)

    else:
        p = figure(title='Horizontal Bar Chart', y_range=df_pie_agg["Parameter"], 
                plot_width=900, plot_height=600,
                y_axis_label='Count',tooltips="@Parameter: @{Count}", tools = TOOLS)

        p.hbar(y="Parameter",height = 0.75, left=0, right= "Count", 
            fill_color='black', line_color='white',source = df_pie_agg)


        
    p.xaxis.major_label_orientation = 1.57
    script, div = components(p)

    path = "temp/" + str(session.get('user_id')) + "-temp.csv"
    df_pie_agg.to_csv(path)
    
    return render_template(
    'graphs/bar_plot.html',
    plot_script=script,
    plot_div=div,
    js_resources=INLINE.render_js(),
    css_resources=INLINE.render_css(),
    graphSelected = True,
    columns = data.columns,
    selected = selected_parameter,
    path=path
    ).encode(encoding='UTF-8')
    

def PCA_transformation(data, reduce_to = None, var_ratio = None):
    """
     Apply Principal Component Analysis (PCA) to Dataset.
    Dataset is first standirtized. Then Dataset is reduced to n = reduce_to-D dimension. 
    If the data has an object column, error should be given. Or assume that it is intended and reduce the only numerical columns

    Parameters:
    :data: -- Dataframe that is given
    :reduce_to: -- The final dimension after PCA transformation
    :var_ratio: -- If given, reduce until
   

    Return:
    :pca: -- PCA class
    :reduced_df: -- Dataframe that is obtained by reducing our initial dataframe to dimension inputted. This dataframe will consist
    of eigenvectors that shares the maximum variance.
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    if reduce_to == None:
        reduce_to = 0.80 # -- default ratio value
    
    # Ready the dataset and variables
    if var_ratio != None:
        if var_ratio <= 1 and var_ratio >= 0:
            reduce_to = var_ratio
        else:
            print("Error..")
            return None


    numerical_df = data.select_dtypes(exclude = ["object"])
    scaler = StandardScaler()
    pca = PCA(n_components=reduce_to)

    # Standirtize
    numerical_df = scaler.fit_transform(numerical_df)

    # PCA Analysis
    pca.fit(numerical_df)
    reduced_df = pca.transform(numerical_df)

    # Return dataframe
    column_names = ["var" + str(i) for i in range(0,reduced_df.shape[1])]
    reduced_df = pd.DataFrame(reduced_df,columns = column_names)
    return reduced_df,pca

def PCA_transformation_describe(new_df,pca):
    """
    Return a line graph by using bokeh. 
    :pca: -- pca module that contains variance ratios
    :new_df: -- df that is newly transformed
    """

    #Create the graph
    p = figure(title = "Variance Ratio", plot_width = 1200,plot_height = 800,
    x_axis_label = "Number of components",y_axis_label = "Cumilative Ratio")
    
    line_y = np.cumsum(pca.explained_variance_ratio_)
    line_x = range(1,len(line_y)+1)
    df_line = pd.DataFrame()
    np.cumsum(pca.explained_variance_ratio_)
    p.line(x = line_x, y = line_y, color = "black", line_width = 2 )
    script, div = components(p)

    return render_template(
    'transformation/pca_transform.html',
    plot_script=script,
    plot_div=div,
    js_resources=INLINE.render_js(),
    css_resources=INLINE.render_css(),
    graphSelected = True,
    ).encode(encoding='UTF-8')
    #Extra information