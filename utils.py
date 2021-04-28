import numpy as np
import pandas as pd
import time
from run import ALLOWED_EXTENSIONS
from flask_wtf import FlaskForm
from collections import OrderedDict
from flask import Flask, request, redirect, url_for,render_template
from bokeh.models import ColumnDataSource, Div, Select, Slider, TextInput
from bokeh.io import curdoc
from bokeh.resources import INLINE
from bokeh.embed import components
from bokeh.plotting import figure, output_file, show
from bokeh.layouts import gridplot
import bisect
from math import pi
from numpy import arange
from itertools import chain
from collections import OrderedDict
from bokeh.palettes import RdBu as colors
from bokeh.models import ColorBar, LinearColorMapper

def load_dataset(path,delimitter,qualifier,assumption = False):
    """
    Read file with given extension.
    If CSV : Read columns with pandas
    if TXT : Read columns manually

    Parameters:
    :path: -- path to the txt file
    :delimitter: -- delimitter for splitting the columns/values
    :assumption: -- if true, assume first line holds the information about valueTypes

    Returns:
    :dataTypes: - return dataTypes list
    :dataColumns: -- return dataColumns list
    :df: -- return dataframe of the file
    """
    if delimitter == "":
        delimitter = ","
    if qualifier == "":
        qualifier = '""'


    extension = path.split(".")[1]
    if(extension == "csv"):
        df = pd.read_csv(path)
        dataTypes = df.dtypes
        dataColumns = df.columns
        return dataTypes,dataColumns,df
    elif(extension == "txt"):
        dataColumns = []
        values = []
        dataTypes = []
        isTypes = assumption
        isColumn = True
        with open(path,mode = "r") as file:
            for line in file:
                # read line and split the columns/types
                line = (line.rstrip('\n')).split(delimitter)
                line = [col.strip(qualifier) for col in line]

                # if value types/columns
                if isTypes:
                    dataTypes = line
                    isTypes = False
                    
                elif isColumn:
                    dataColumns = line
                    isColumn = False

                # if values
                else:
                    values += [line]
        # dataframe is created
        df = pd.DataFrame(data = values, columns = dataColumns)
        #df.set_index(df.columns[0]) :>
        #df.drop([df.columns[0]],inplace=True,axis=1)
        if assumption:
            df = assign_datatypes(df,dataTypes)
        else:
            dataTypes = df.dtypes

        return dataTypes,dataColumns,df

#Check if file is valid
def allowed_file(filename):
    """
    Check if the file is valid.
    
    Parameters:
    :filename: -- path to the file.

    Return true if extension is correct.
    """
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


def assign_datatypes(data,dataTypes):
    """
    Change the datatypes in the columns of data with respect to the dataTypes
    Parameters:
    :data: -- given dataframe whose dataTypes are all strings
    :dataTypes: -- given list of data types that determines the types of the data
    """
    for i in range(len(dataTypes)):
        data.iloc[:,i] = data.iloc[:,i].astype(dataTypes[i])
    return data

def choose_attributes(df,attributes):
    """
    Choose the attributes and return the new chosen dataframe.
    Parameters:
    :df: -- given dataframe 
    :attributes: -- features/columns that is picked by the user
    """
    chosen_df = df[attributes]
    unchosen_df = df.drop(attributes,axis=1)
    return chosen_df,unchosen_df

def groupColumns(df):
    dtypeArr = []
    columnArr = []
    lens = []
    returnArr = []
    type_dct = {str(k): list(v) for k, v in df.groupby(df.dtypes, axis=1)}
    type_dct = OrderedDict(sorted(type_dct.items(), key=lambda i: -len(i[1])))

    for types in type_dct:
        type_dct[types].sort()
        columnArr.append(type_dct[types])
        dtypeArr.append(types)
        lens.append(len(type_dct[types]))
    
    for i in range(max(lens)):
        arr = []
        for k in range(len(dtypeArr)):
            if(i < lens[k]):
                arr.append(columnArr[k][i])
        returnArr.append(arr)
    return dtypeArr, returnArr


def scatter_plot(df,x_name,y_name):
#create the graph
    fig = figure(plot_height=600, plot_width=720)
    source = ColumnDataSource()
    fig.circle(x="x", y="y", source=source, size=5,line_color=None)

    #choose x and y label for plotting
    fig.xaxis.axis_label = x_name
    fig.yaxis.axis_label = y_name
    source.data = dict(
        x = df[x_name],
        y = df[y_name]
    )
    script, div = components(fig)
    #return template
    return render_template(
        'scatter_plot.html',
        plot_script=script,
        plot_div=div,
        js_resources=INLINE.render_js(),
        css_resources=INLINE.render_css(),
        x_name = x_name,
        y_name = y_name,
        graphSelected = True,
        columns = df.columns
    ).encode(encoding='UTF-8')

def scatter_matrix(dataset,features):
    dataset_selected = dataset[features]
    dataset_source = ColumnDataSource(data=dataset)
    scatter_plots = []
    y_max = len(dataset_selected.columns)-1
    for i, y_col in enumerate(dataset_selected.columns):
        for j, x_col in enumerate(dataset_selected.columns):
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
    TOOLS = "box_select,lasso_select,pan,wheel_zoom,box_zoom,reset,help"
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

    script, div = components(p)
    return render_template(
    'graphs/correlation_plot.html',
    plot_script=script,
    plot_div=div,
    js_resources=INLINE.render_js(),
    css_resources=INLINE.render_css(),
    graphSelected = True,
    columns = df.columns
    ).encode(encoding='UTF-8')

def pie_plot(data,selected_parameter, sort_by_values = False):
    from math import pi 
    from bokeh.transform import cumsum 
    from bokeh.palettes import inferno
    df_pie_agg = pd.DataFrame()
    
    objects_data = data[selected_parameter].unique()
    val_count = []

    for vals in objects_data:
        val_count += [data[data[selected_parameter] == vals].shape[0]]
    print(objects_data,val_count)
    df_pie_agg = pd.DataFrame()
    df_pie_agg["Parameter"] = objects_data
    df_pie_agg["Count"] = val_count
    print(val_count)
    # Add angles based on Win Count so each wedge is the right size
    df_pie_agg['Angle'] = df_pie_agg['Count']/df_pie_agg['Count'].sum() * 2*pi
    df_pie_agg['color'] = inferno(len(val_count))
    
    if sort_by_values:
        df_pie_agg = df_pie_agg.sort_values(by = 'Count', ascending = False)

    # Draw a chart
    p = figure(title='Pie Chart', x_range=(-0.5, 1.0),
            plot_width=800, plot_height=600, 
            toolbar_location=None, tooltips="@Parameter: @{Count}") 

    p.wedge(x=0.1, y=1, radius=0.4,
            start_angle=cumsum('Angle', include_zero=True), 
            end_angle=cumsum('Angle'),
            line_color="white", 
            legend='Parameter', 
            fill_color = 'color',
            source=df_pie_agg)

    # These options can be used to make the chart cleaner
    #p.axis.axis_label=None
    #p.axis.visible=False
    #p.grid.grid_line_color = None
    #p.outline_line_color = None

    script, div = components(p)
    return render_template(
    'graphs/pie_plot.html',
    plot_script=script,
    plot_div=div,
    js_resources=INLINE.render_js(),
    css_resources=INLINE.render_css(),
    graphSelected = True,
    selected = selected_parameter,
    columns = data.columns
    ).encode(encoding='UTF-8')
    return "sex"

def dist_plot(df,parameter,bins = 20):
    
    # Use numpy to create histogram bins
    hist, edges = np.histogram(df[parameter], bins=bins)

    # Draw a chart
    p = figure(title='Histogram', plot_width=800, plot_height=600,
            x_axis_label=parameter, y_axis_label='Count')

    p.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:], line_color='white', fill_color='black')


    script, div = components(p)
    return render_template(
    'graphs/dist_plot.html',
    plot_script=script,
    plot_div=div,
    js_resources=INLINE.render_js(),
    css_resources=INLINE.render_css(),
    graphSelected = True,
    columns = df.columns,
    selected = parameter
    ).encode(encoding='UTF-8')

def nan_plot(df):
    return "Show Nan"


def bar_plot(data,selected_parameter, option = 'Vertical'):
    # Draw a chart
    df_pie_agg = pd.DataFrame()
    
    data = data.dropna(subset=[selected_parameter])
    objects_data = data[selected_parameter].unique()
    val_count = []
    for vals in objects_data:
        val_count += [data[data[selected_parameter] == vals].shape[0]]

    df_pie_agg = pd.DataFrame()
    df_pie_agg["Parameter"] = objects_data
    df_pie_agg["Count"] = val_count
    print(objects_data,val_count)

    if option == 'Vertical':
        p = figure(title='Vertical Bar Chart', x_range=df_pie_agg["Parameter"], 
                plot_width=900, plot_height=600,
                y_axis_label='Count')

        p.vbar(x=df_pie_agg["Parameter"], width = 0.75, bottom=0, top= df_pie_agg["Count"], 
            fill_color='black', line_color='white')

    else:
        p = figure(title='Horizontal Bar Chart', y_range=df_pie_agg["Parameter"], 
                plot_width=900, plot_height=600,
                y_axis_label='Count')

        p.hbar(y=df_pie_agg["Parameter"],height = 0.75, left=0, right= df_pie_agg["Count"], 
            fill_color='black', line_color='white')


        
    p.xaxis.major_label_orientation = 1.57
    script, div = components(p)

    return render_template(
    'graphs/bar_plot.html',
    plot_script=script,
    plot_div=div,
    js_resources=INLINE.render_js(),
    css_resources=INLINE.render_css(),
    graphSelected = True,
    columns = data.columns,
    selected = selected_parameter
    ).encode(encoding='UTF-8')
