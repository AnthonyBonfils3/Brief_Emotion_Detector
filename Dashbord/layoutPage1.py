# Dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table


#################################################################################################################
################################################## LAYOUT #######################################################
#################################################################################################################
layoutPage1 = html.Div([
    html.Header([
        dcc.Link(html.Button('Go to Home Page', className='pth_button'), href='/'),
        dcc.Link(html.Button('Classifications results', className='pth_button'), href="/Classifications%20Results"),
        html.Br(),
        html.H1('Emotions detector'),
        html.H2('Datas Analysis'),

    ]),
    html.Aside(

    ),
    html.Tbody(id='main_block',children=[
        html.Div(id='Block_left', children=[
            html.Article(id='left_selector',children=[

                ## Data Selector
                html.H3('Select a DataSet'),
                dcc.Dropdown(
                    id='DataSet_dropdown',
                    options=[
                        {'label': 'First one : Kaggle', 'value': 'Emotion_final.csv'},
                        {'label': 'Second one : Data.word', 'value': 'text_emotion.csv'},
                    ],
                    optionHeight= 60,
                    value='Emotion_final.csv',
                    clearable=False,
                ),

                ## Emotion selector
                html.H3('Select an emotion'),
                dcc.RadioItems(
                   id='Emotion_radio'
                   ),
                html.H3('Emotions histogram'),

                ## Fig 1 : Histigramme Emotions
                dcc.Graph(
                    id='Hist_emotions'
                    ),
            ]),
        ]),
        html.Div(id='Block_right', children=[
            html.Section(id='Block_1',children=[
                html.Article(id='Block_1_Article_1', children=[

                    ## Fig 2 : Histigramme Mots
                    html.H3("Words ordered by rank. The first rank is the most frequent words and the last one is the less present"),
                    dcc.Graph(
                        id='Hist_mots',
                        ),
                    dcc.RangeSlider(
                        id='word_rank_slider',
                        min=0,
                        max=100,
                        step=1,
                        value=[2, 50],
                        marks={
                            0: {'label': 'Top(min)', 'style': {'color': '#77b0b1'}},
                            50: {'label': 'Top(50)'},
                            100: {'label': 'Top(Max)', 'style': {'color': '#77b0b1'}}
                        },
                        allowCross=False
                    ),
                ]),

                html.Article(id='Block_1_Article_2', children=[
                    html.H3('Emotions Repartition'),
                    dcc.Graph(
                        # Fig 3 : Pie Chart
                        id='Pie_chart'
                        ),
                ]),
            ]),
            html.Section(id='Block_2',children=[
                html.Article(id='Block_2_Article_1', children=[
                    html.H3('Datas Table'), 
                    # Tableau des données
                    html.Div(id='div_table_data',children=[
                        html.Div(id='page1_table'),
                    ]),
                ]),
            ]),        
        ]),
    ]),

    html.Footer([
        html.Br(),
        dcc.Link(html.Button('Go to Home Page', className='pth_button'), href='/'),
        dcc.Link(html.Button('Classifications results', className='pth_button'), href="/Classifications%20Results")     
    ]),
])