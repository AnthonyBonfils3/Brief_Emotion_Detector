import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import dash_table

dfk = pd.read_csv("./Datas/Emotion_final.csv")

layoutPage2 = html.Div([
    html.Header([
        html.H1('Emotion detector'),
        html.H2('Classification Results'),

    ]),
    html.Aside(

    ),
    html.Tbody(id='main_block',children=[
        html.Div(id='Block_left', children=[
            html.Article(id='left_selector',children=[

            ]),
        ]),
        html.Div(id='Block_right', children=[
            html.Section(id='Block_1',children=[
                html.Article(id='Block_1_Article_1', children=[

                ]),
                html.Article(id='Block_1_Article_2', children=[

                ]),
            ]),
            html.Section(id='Block_2',children=[
                html.Article(id='Block_2_Article_1', children=[
                    
                ]),
            ]),        
        ]),
    ]),

    html.Footer([
    html.Div(id='app-2-display-value'),
    html.Br(),
    dcc.Link('Go to Home Page', href='/'),
    html.Br(),
    dcc.Link('Datas analysis', href='/Datas%20Analysis') 
    ]),
])
