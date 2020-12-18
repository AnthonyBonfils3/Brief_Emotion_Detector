# Dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import dash_table

# Classiques
import pandas as pd
import numpy as np
from collections import defaultdict
import pickle

# plotly
import plotly.express as px
import plotly.graph_objects as go

### evaluation
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn import metrics



## models : pickles
filename = "pipes_models.pickle"
pipes = pickle.load(open(filename, 'rb'))

## Datas
dfk = pd.read_csv("./Datas/Emotion_final.csv")
emot = dfk.Emotion.unique()
corpusk = dfk.Text
targetsk = dfk.Emotion
targetsk = np.array([1 if x == emot[0] else 2 if x==emot[1] else 3 if x==emot[2] else 4 if x==emot[3] else 5 if x==emot[4] else 6 for x in targetsk])


## Récup les perfs des modèles
res=defaultdict(list)
for pipe in pipes:
    # name of the model
    name = "-".join([x[0] for x in pipe.steps])

    # predict and save results
    y = pipe.predict(corpusk)
    res[name].append([
        pipe.steps[2][0],
        f1_score(targetsk, y, average='micro'),
        f1_score(targetsk, y, average='macro'),
        f1_score(targetsk, y, average='weighted'),
        precision_score(targetsk, y, average='weighted'), 
        recall_score(targetsk, y, average='weighted')
    ])

## Genere la Table des Résultats
def print_table_res(res):
    # Compute mean and std
    final = {}
    for model in res:
        final[model] = {
            "name":res[model][0][0],
            "f1_av_micro (%)": res[model][0][1].round(3)*100,
            "f1_av_macro (%)": res[model][0][2].round(3)*100,
            "f1_av_weighted (%)": res[model][0][3].round(3)*100,
            "prec_av_weighted (%)": res[model][0][3].round(3)*100,
            "recall_av_weighted (%)": res[model][0][3].round(3)*100,
        }
    df = pd.DataFrame.from_dict(final, orient="index").round(3)
    return df
df_res = print_table_res(res)


########################################################################################
#################################### LAYOUT ############################################
########################################################################################
layoutPage2 = html.Div([
    html.Header([
        dcc.Link(html.Button('Go to Home Page', className='pth_button'), href='/'),
        dcc.Link(html.Button('Datas analysis', className='pth_button'), href='/Datas%20Analysis'),
        html.Br(),
        html.H1('Emotions detector'),
        html.H2('Classification Results'),

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

                ## Model Selector
                html.H3('Select a model'),
                dcc.RadioItems(
                    id='solver_radio',
                    options=[{'label': 'Stochastic Gradiant Decent', 'value': 0},
                        {'label': 'Linear SVM', 'value': 1},
                        {'label': 'Logistic Regression', 'value': 2},
                        {'label': 'Decision Tree', 'value': 3},
                        {'label': 'Complément NB', 'value': 4},
                        ],
                    value = 0,
                    ),
            ]),
        ]),
        html.Div(id='Block_right', children=[
            html.Section(id='Block_1',children=[

                ## Correlation Matrix
                html.Article(id='Block_1_Article_1', children=[
                    html.H3('Correlation Matrix'),
                    dcc.Graph(id='corr_matrix', 
                        config={'responsive':True}
                    ), 
                ]),
                html.Article(id='Block_1_Article_2', children=[
                    
                    ## ROC Curve or Precision recall
                    dbc.Tabs(
                        [
                            dbc.Tab(label="ROC Curves", tab_id="ROC"),
                            dbc.Tab(label="Precision-Recall Curves", tab_id="prec_recall", activeTabClassName='Tab'
                            ),
                        ],
                        id="tabs_fig",
                        active_tab="ROC",
                    ),
                    html.Div(id="tab_content"),
                ]),
            ]),
            html.Section(id='Block_2',children=[
                html.Article(id='Block_2_Article_1', children=[

                    ## Tableau récap modèles
                    html.H3('Predictions results for differents models'), 
                    html.Div(id='div_table_res',children=[
                        dash_table.DataTable(
                            id='app_2_table',
                            export_format='csv',
                            export_headers='display',
                            columns=[{'id': c, 'name': c} for c in df_res.columns],
                            data= df_res.to_dict('records'),
                            style_as_list_view=True,
                            fixed_rows={'headers': True},
                            #fixed_columns={'headers': True, 'data' :1},# garder les names quand on scroll : le chiffre correspond à l'indice de la colonne
                            style_table={
                                'overflowX': 'auto',
                                'overflowY': 'auto',
                                'maxHeight':'400px',
                                'maxWidth':'1000px'},
                            #Cell dim + textpos
                            style_cell_conditional=[{
                                'height': 'auto',
                                # all three widths are needed
                                'minWidth': '100px', 'width': '120px', 'maxWidth': '300px',
                                'whiteSpace': 'normal','textAlign':'center',
                                'backgroundColor': '#1e2130',
                                'color': 'white'
                                }],
                            #Line strip
                            style_data_conditional=[{
                                'if': {'row_index': 'odd'},
                                'backgroundColor': '#161a28',
                                'color': 'white'
                                }],
                            style_header={
                                'backgroundColor': 'rgb(50, 50, 50)',
                                'fontWeight': 'bold',
                                'color':'white'},
                            # Tool Tips
                            tooltip_data=[{
                                column: {'value': str(value), 'type': 'markdown'} for column, value in row.items()
                                } for row in df_res.to_dict('rows')],
                            tooltip_duration=None
                            ),
                        ]),
                ]),
            ]),        
        ]),
    ]),

    html.Footer([
    html.Div(id='app-2-display-value'),
    html.Br(),
    dcc.Link(html.Button('Go to Home Page', className='pth_button'), href='/'),
    dcc.Link(html.Button('Datas analysis', className='pth_button'), href='/Datas%20Analysis') 
    ]),
])
