import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import dash_table

df = pd.read_csv("./Datas/Emotion_final.csv")

layoutHome = html.Div([
    html.H3('HomePage'),
    dcc.Dropdown(id='app-home-dropdown'),  
    html.Div(id='app-home-display-value'),
    dcc.Link('Table des données TimesData', href='/apps/app1'),
    html.Br(),
    dcc.Link("Cas d'étude.", href='/apps/app2')
])

