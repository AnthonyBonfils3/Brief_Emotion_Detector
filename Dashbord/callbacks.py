from dash.dependencies import Input, Output, State
from app import app

import dash_table

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from collections import defaultdict
# sklearn
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
import pickle


dfk = pd.read_csv("./Datas/Emotion_final.csv")
emot = dfk.Emotion.unique()
corpusk = dfk.Text
targetsk = dfk.Emotion
targetsk = np.array([1 if x == emot[0] else 2 if x==emot[1] else 3 if x==emot[2] else 4 if x==emot[3] else 5 if x==emot[4] else 6 for x in targetsk])

# import models as pickles
filename = "pipes_models.pickle"
pipes = pickle.load(open(filename, 'rb'))

# Sub-sample the data to plot. take the 50 first + the rest sample with the given step 
def subsample(x, init, end, step=400):
    return np.hstack((x[init:end], x[end:end:step]))


#############################################################################
############################### PAGE HOME ###################################
#############################################################################
@app.callback(
    Output("modal_Home", "is_open"),
    [Input("open-centered", "n_clicks"), Input("close-centered", "n_clicks")],
    [State("modal_Home", "is_open")],
)
def toggle_modal(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open

@app.callback(
    Output('app-home-display-value', 'children'),
    Input('app-home-dropdown', 'value'))
def display_value(value):
    return 'You have selected "{}"'.format(value)

#############################################################################
################################# PAGE 1 ####################################
#############################################################################
@app.callback(
    Output('page1_table','children'), 
    Input('Emotion_radio', 'value'))
def display_table_page1(Emotion_value):
    if Emotion_value == 'all':
        df = dfk
    else:
        df=dfk.loc[dfk.Emotion==Emotion_value]

    table = dash_table.DataTable(
        id='app-1-table',
        columns=[{'id': c, 'name': c} for c in df.columns],
        data= df.to_dict('records'),
        style_as_list_view=True,
        fixed_rows={'headers': True},
        style_table={
            'overflowX': 'auto',
            'overflowY': 'auto',
            'maxHeight':'400px',
            'maxWidth':'1600px'},
        #Cell dim + textpos
        style_cell_conditional=[
            #{'if': {'column_id': 'Emotion'},'width': '15%'},
            {'if': {'column_id': 'Text'},'width': '50%'},{
            'height': 'auto',
            # all three widths are needed
            'minWidth': '180px', 'width': '180px', 'maxWidth': '180px',
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
            } for row in df.to_dict('rows')],
        tooltip_duration=None
        ),  

    return table


@app.callback(
    Output('Hist_mots','figure'), 
    Input('Emotion_radio', 'value'),
    [Input('word_rank_slider', 'value')])
def display_hist_mots_page1(Emotion_value, slid_value):
    if Emotion_value == 'all':
        df = dfk
    else:
        df=dfk.loc[dfk.Emotion==Emotion_value]

    vect = CountVectorizer(stop_words='english')#stop_words=stopwords
    X = vect.fit_transform(df.Text)
    words = vect.get_feature_names()

    # Compute rank
    wsum = np.array(X.sum(0))[0]
    ix = wsum.argsort()[::-1]
    wrank = wsum[ix] 
    labels = [words[i] for i in ix]

    trace = go.Bar(x = subsample(labels, slid_value[0], slid_value[1]), 
                   y = subsample(wrank, slid_value[0], slid_value[1]),
                   marker = dict(color = 'rgba(255, 174, 255, 0.5)',
                   line = dict(color ='rgb(0,0,0)',width =1.5)),
    )
    layout = go.Layout(
                    xaxis_title_text = 'Word rank',
                    yaxis_title_text = 'word frequency',
                    paper_bgcolor = 'rgb(22,26,40)',
                    plot_bgcolor = 'rgb(22,26,40)',
                    font_color='white')
    figure = go.Figure(data = trace, layout = layout)
    return figure
