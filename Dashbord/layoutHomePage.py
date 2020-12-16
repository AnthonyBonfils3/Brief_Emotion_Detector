import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import pandas as pd
import dash_table
from app import app
from my_imports.my_var import modal

df = pd.read_csv("./Datas/Emotion_final.csv")

layoutHome = html.Div(id='app-homePage', children=[

    html.H1('Emotions Detector'),
    modal,
    html.H2('Welcome to the API who analyse emotions in your sentences'),
    dcc.Dropdown(id='app-home-dropdown'),  
    html.Div(id='app-home-display-value'),
    html.Div(id='div_roue', children=[
        html.Img(id='img_roue', src=app.get_asset_url('/Images/roue_des_emotions.png')),
        ]),
    dcc.Link(html.Button('Datas analysis', className='pth_button'), href='/Datas%20Analysis'),
    dcc.Link(html.Button('Classifications results', className='pth_button'), href="/Classifications%20Results"),
])

