# %%
# For data analysis
import log
import numpy as np
import pandas as pd
import folium
import dash_table
from folium.plugins import HeatMap
import plotly.graph_objs as go
# For model creation and performance evaluation
# For visualizations and interactive dashboard creation
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
import warnings
import dash_leaflet as dl
import base64
warnings.filterwarnings('ignore')

airbnb=pd.read_csv('AB_NYC_2019.csv')
print(airbnb.shape)
print(airbnb.dtypes)
print(airbnb.dtypes)
print(airbnb.head())
#Removing the Duplicates if any
airbnb.duplicated().sum()
airbnb.drop_duplicates(inplace=True)

print(airbnb.isnull().sum())
airbnb.drop(['name','id','host_name','last_review'], axis=1, inplace=True)

print(airbnb.head(5))
#Rreplace the 'reviews per month' by zero
airbnb.fillna({'reviews_per_month':0}, inplace=True)
#examing changes
#print(airbnb.reviews_per_month.isnull().sum())

#Remove the NaN values from the dataset
airbnb.isnull().sum()
airbnb.dropna(how='any',inplace=True)
print(airbnb.info())
image_path = "map-of-new-york-city.jpg"
encoded_image = base64.b64encode(open(image_path, 'rb').read())
image_path1 = "FFFig.png"
encoded_image1 = base64.b64encode(open(image_path1, 'rb').read())
image_path2 = "1.png"
encoded_image2 = base64.b64encode(open(image_path2, 'rb').read())
def correlation_plot():
    fig = px.scatter(airbnb, x='longitude', y='latitude', color='neighbourhood_group', color_discrete_sequence=px.colors.sequential.GnBu)
    fig.update_layout(title="Correlation Plot")
    return fig


airbnb_numeric = airbnb.select_dtypes(include=['float64', 'int64'])
fig = go.Figure(data=go.Heatmap(z=airbnb_numeric.corr(),x=airbnb_numeric.columns,  # 设置 x 轴标签为 airbnb_numeric 的列名
                               y=airbnb_numeric.columns,colorscale='Reds',
                               colorbar=dict(title='Correlation'),
                               zmin=-1, zmax=1,
                               hovertext=airbnb_numeric.corr().round(2)))  # 显示相关性数值，))
fig.update_layout(title="Correlation Heatmap")


fig1 = px.scatter(airbnb, x='longitude', y='latitude', color='neighbourhood_group',
                 template='plotly_dark', color_continuous_scale='GnBu')
fig1.update_layout(title="Scatter Plot of Latitude and Longitude",
                  xaxis_title="Longitude",
                  yaxis_title="Latitude")
# Create the Dash app
app = dash.Dash(__name__)
server = app.server

# create data table
table_style = { 'height': '500%','overflowY': 'scroll','width': '55%', 'overflowX': 'scroll','bottom': '0px','height': '2000px',
    'left': '710px'}

# Create example app.
options_minimum_nights = [{'label': str(i), 'value': i} for i in range(366)]
options_number_of_reviews = [{'label': str(i), 'value': i} for i in range(1000)]
options_reviews_per_month= [{'label': str(i), 'value': i} for i in range(1000)]
options_availability_365 = [{'label': str(i), 'value': i} for i in range(366)]
options_calculated_host_listings_count= [{'label': str(i), 'value': i} for i in range(1000)]

# Define the layout of the dashboard
app.layout = html.Div(
    style={'font-family': 'Arial, sans-serif', 'max-width': '1600px','margin-top': '20px',
           'margin': '0 auto', 'padding': '2px', 'background-color': '#F0F0F0'},
    children=[
        html.H1('COMP7103 Lab: NY city rental house price Prediction',
                style={'text-align': 'center', 'color': '#333333'}),
         html.H1('Author : YANG Jingwen + LI Zixi',
                style={'text-align': 'center', 'color': '#333333'}),
        # Layout for exploratory data analysis: correlation between two selected features
        html.Div([html.H1('Map of New York City'),html.Img(src='data:image/png;base64,{}'.format(encoded_image.decode()), style={'width': '100%'})]),
        
      html.Div(
                style={'display': 'flex', 'justify-content': 'space-between'},
                children=[
                    html.H1('HEAPMAP of New York City', style={'color': '#333333'}),
                    html.H1('New York Airbnb', style={'color': '#333333'})]),
    html.Div(
    children=[
        html.Img(style={'position': 'absolute',
                'top': '2150px',
                'right': '200px',
                'transform': 'scale(1.6)'
                },
            src='data:image/png;base64,{}'.format(encoded_image2.decode()))]),
        html.Div(children=[
    html.Div(children=[dcc.Graph(id='heatmap', figure=fig),
                       html.Div(style={'width': '150px'}),  # 添加间距,
        dcc.Graph(id='scatter-plot', figure=fig1)
    ], style={'display': 'flex'}),
]),
    html.Div(
    style={'display': 'flex', 'justify-content': 'space-between'},
    children=[
        html.H1('Exploratory Data Analysis --Visualization', style={'color': '#333333'}),
        html.H1('Part of Old Data display', style={'color': '#333333'})]),
        html.Div(style={'height': '100px'},
    className='scroll-table',
    children=[
        dash_table.DataTable(
            style_table=table_style,
            id='data-table',
            columns=[{'name': col, 'id': col} for col in airbnb.columns],
            data=airbnb.head(200).to_dict('records'),
            fixed_rows={'headers': True, 'data': 0}, 
            style_cell={'minWidth': '100px', 'width': '100px', 'maxWidth': '100px'},  
            virtualization=False  
            )]),      
        
        
        html.Div([
            html.Label('Correlation between Feature (X-axis) and price', style={'color': '#777777','margin-top': '-250px'}),
            dcc.Dropdown(
                id='x_feature',
                options=[{'label': col, 'value': col} for col in airbnb.columns],
                value=airbnb.columns[0],
                style={'width': '100%', 'background-color': '#FFFFFF','margin-top': '0px'}
            )
        ], style={'width': '30%', 'margin-right': '20px','margin-top': '-120px'}),
         
        dcc.Graph(id='correlation_plot',style={'height': '400px', 'width': '700px','margin-top': '20px','margin-top': '20px'}),
        
        # Layout for wine quality prediction based on input feature values
        html.H3("NY City rental house price Prediction", style={'bottom': '-500px','left': '650px' ,'color': '#555555'}),
        html.Div([
            html.Label("minimum_nights", style={'color': '#777777'}),html.Br(),
            dcc.Dropdown(
                    id='minimum_nights',
                    options=options_minimum_nights,
                    style={'width': '30%', 'background-color': '#FFFFFF'}
                ),
            html.Br(),
            html.Label("number_of_reviews", style={'color': '#777777'}),html.Br(),
            dcc.Dropdown(
                    id='number_of_reviews',
                    options=options_number_of_reviews,
                    style={'width': '30%', 'background-color': '#FFFFFF'}
                ),
            html.Br(),
            html.Label("reviews_per_month", style={'color': '#777777'}),html.Br(),
            dcc.Dropdown(
                    id='reviews_per_month',
                    options=options_reviews_per_month,
                    style={'width': '30%', 'background-color': '#FFFFFF'}
                ),
            html.Br(),
            html.Label("calculated_host_listings_count", style={'color': '#777777'}),html.Br(),
             dcc.Dropdown(
                    id='calculated_host_listings_count',
                    options=options_calculated_host_listings_count,
                    style={'width': '30%', 'background-color': '#FFFFFF'}
                ),
            html.Br(),
            html.Label("availability_365", style={'color': '#777777'}),html.Br(),
            dcc.Dropdown(
                    id='availability_365',
                    options=options_availability_365,
                    
                    style={'width': '30%', 'background-color': '#FFFFFF'}
                ),
            html.Br(),
            html.Label("neighbourhood_group", style={'color': '#777777'}),html.Br(),
            dcc.Dropdown(
                    id='neighbourhood_group',
                    options=[
                        {'label': 'Brooklyn', 'value': 'Brooklyn'},
                        {'label': 'Manhattan', 'value': 'Manhattan'},
                        {'label': 'Queens', 'value': 'Queens'},
                        {'label': 'Staten Island', 'value': 'Staten Island'},
                        {'label': 'Bronx', 'value': 'Bronx'},
                    ],
                    
                    style={'width': '30%', 'background-color': '#FFFFFF'}
                ),
            html.Br(),

            html.Label("room_type", style={'color': '#777777'}),html.Br(),
            dcc.Dropdown(
    id='room_type',
    options=[
        {'label': 'Shared room', 'value': 'Shared room'},
        {'label': 'Private room', 'value': 'Private room'},
        {'label': 'Entire home/apt', 'value': 'Entire home/apt'}],
    
    style={'width': '30%', 'background-color': '#FFFFFF'}
),
            html.Br(),
        ],style={'bottom': '-500px','left': '650px' ,'color': '#555555'}),
        
        html.Div([
            html.Button('Predict', id='predict-button',
                        n_clicks=0, style={'margin-top': '0px', 'background-color': '#333333', 'color': '#FFFFFF'}),
        ]),
        html.Div([
            html.H3("Prediction:",style={'font-weight': 'bold', 'font-size': '18px'}),
            html.Div(id='prediction-output',
                     style={'font-weight': 'bold', 'font-size': '18px'}),
  
        ]),
        html.Div([html.H1('WordCloud of New York City'),
                   html.Img(src='data:image/png;base64,{}'.format(encoded_image1.decode()), style={'width': '100%'})]),
    ],
    
)


# %%
# Define the callback to update the correlation plot

@app.callback(
    dash.dependencies.Output('correlation_plot', 'figure'),
    [dash.dependencies.Input('x_feature', 'value')]
)
def update_correlation_plot(x_feature):
    fig = px.scatter(airbnb, x=x_feature, y='price')
    fig.update_layout(title=f"Correlation between {x_feature} and price")
    return fig

# Define the callback function to predict wine quality


@app.callback(
    Output(component_id='prediction-output', component_property='children'),
    [Input('predict-button', 'n_clicks')],
    [State('minimum_nights', 'value'),
     State('number_of_reviews', 'value'),
     State('reviews_per_month', 'value'),
     State('calculated_host_listings_count', 'value'),
     State('availability_365', 'value'),
     State('neighbourhood_group', 'value'),
     State('room_type', 'value')]
)
def predict_quality(n_clicks, minimum_nights, number_of_reviews, reviews_per_month,
                    calculated_host_listings_count,availability_365,neighbourhood_group, room_type):
    input_features = [minimum_nights, number_of_reviews, reviews_per_month,calculated_host_listings_count,availability_365, neighbourhood_group,room_type]
    
    columns = ['minimum_nights', 'number_of_reviews', 'reviews_per_month',
            'calculated_host_listings_count', 'availability_365',
            'neighbourhood_group_Brooklyn', 'neighbourhood_group_Manhattan',
            'neighbourhood_group_Queens', 'neighbourhood_group_Staten Island',
            'room_type_Private room', 'room_type_Shared room']

    df = pd.DataFrame(columns=columns)
    new_row = [0,0,0,0,0,0,0,0,0,0,0]
    df.loc[0] = new_row
    df.loc[0, 'minimum_nights'] = input_features[0]
    df.loc[0, 'number_of_reviews'] =input_features[1]
    df.loc[0, 'reviews_per_month'] =  input_features[2]
    df.loc[0, 'calculated_host_listings_count'] = input_features[3]
    df.loc[0, 'availability_365'] = input_features[4]
    if neighbourhood_group=='Brooklyn':
        df.loc[0, 'neighbourhood_group_Brooklyn'] = 1
    elif neighbourhood_group=='Manhattan':
        df.loc[0, 'neighbourhood_group_Manhattan'] = 1
    elif neighbourhood_group=='Queens':
        df.loc[0, 'neighbourhood_group_Queens'] = 1
    elif neighbourhood_group=='Staten Island':
        df.loc[0, 'neighbourhood_group_Staten Island'] = 1    

    if room_type=='Private room':
        df.loc[0, 'room_type_Private room'] = 1
    elif room_type=='Shared room':
        df.loc[0, 'room_type_Shared room'] = 1
    y_pred = log.LOG(df)
    prediction =  y_pred

    # Return the prediction
    return html.Div(
        children=[
            html.H2(f"{prediction[0]} $ per night", style={'font-size': '30px', 'color': 'blue', 'font-weight': 'bold','margin-left': '100px'
                                                       ,'margin-top': '-3px'})
        ]
    )

# %%
if __name__ == '__main__':
    app.run_server(debug=False)
