import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import dash_table
from app import app

df = pd.read_csv("./Datas/Emotion_final.csv")

layoutHome = html.Div(id='app-homePage', children=[
    html.H1('Emotions Detector'),
    dcc.Dropdown(id='app-home-dropdown'),  
    html.Div(id='app-home-display-value'),
    dcc.Link('Datas analysis', href='/Datas%20Analysis'),
    html.Br(),
    dcc.Link("Classifications results", href="/Classifications%20Results"),
    html.Div(id='div-roue', children=[
        html.Img(id='img-roue', src=app.get_asset_url('/Images/roue_des_emotions.png')),
        ]),
])

