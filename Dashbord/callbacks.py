#############################################################################
################################# IMPORT ####################################
#############################################################################
from app import app

# Dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_table

# Classique
import pandas as pd
import numpy as np
from collections import defaultdict
import pickle

# Plot
import plotly.graph_objects as go

# sklearn
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn import metrics
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import precision_recall_curve, average_precision_score


#############################################################################
################################# DONNEES ###################################
#############################################################################
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

all_options = {
    'Emotion_final.csv': ['all', 'sadness', 'anger', 'love', 'surprise', 
                          'fear', 'happy'],
    'text_emotion.csv': [u'all', 'empty', 'sadness', 'enthusiasm', 'neutral', 'worry', 'surprise', 'love', 'fun', 'hate', 'happy', 'boredom', 'relief', 'anger']
}




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

@app.callback(
    Output('output_container', 'children'),
    [Input('submit_button', 'n_clicks')],
    [State('text_box_home', 'value')])
def update_output(n_clicks, text):
    if n_clicks > 0:
        text = [text]
        pred= pipes[0].predict(text)

        #emotion = 'sadness'
        #pred= pipes[0].predict([text])
        emotion = emot[pred-1]
        return  u'You have entered: \n{}. Your text seams to be an expression of {}.'.format([text], emotion)




#############################################################################
################################# PAGE 1 ####################################
#############################################################################

@app.callback(
    Output('Emotion_radio', 'options'),
    Input('DataSet_dropdown', 'value'))
def set_emotions_options(selected_dataset):
    return [{'label': i, 'value': i} for i in all_options[selected_dataset]]

@app.callback(
    Output('Emotion_radio', 'value'),
    Input('Emotion_radio', 'options'))
def set_emotions_value(available_options):
    return available_options[0]['value']


# Histigramme des émotions
@app.callback(
    Output('Hist_emotions','figure'), 
    Input('DataSet_dropdown', 'value'))
def display_hist_emotions_page1(dataset):
    df = pd.read_csv('./Datas/'+dataset)
    if dataset == 'text_emotion.csv':
        df.columns = ["Tweet_id", "Emotion", "Author", "Text"]
        df = df.drop(columns='Tweet_id')
        df = df.drop(columns='Author')
        df.loc[df.Emotion =='happiness', 'Emotion'] = 'happy'
    
    fig_Hist_Emot = go.Figure(
        data=[go.Histogram(x=df.Emotion, 
                           name='words count', 
                           ), 
        go.Histogram(x=df.Emotion, 
                     cumulative_enabled=True, 
                     name='cumulative <br>words count', 
                     )],
        layout ={
            'xaxis_title_text': 'Emotions',
            'paper_bgcolor':'rgb(22,26,40)',
            'plot_bgcolor':'rgb(22,26,40)',
            'font_color':'white',
            'legend' : {
                'yanchor':"top",
                'y':1.2,
                'xanchor':"left",
                'x':0.01
                }
            })
    return fig_Hist_Emot


# Histograme mots
@app.callback(
    Output('Hist_mots','figure'), 
    Input('Emotion_radio', 'value'),
    Input('word_rank_slider', 'value'),
    Input('DataSet_dropdown', 'value'))
def display_hist_mots_page1(Emotion_value, slid_value, dataset):
    df_temp = pd.read_csv('./Datas/'+dataset)
    if dataset == 'text_emotion.csv':
        df_temp.columns = ["Tweet_id", "Emotion", "Author", "Text"]
        df_temp = df_temp.drop(columns='Tweet_id')
        df_temp = df_temp.drop(columns='Author')
        df_temp.loc[df_temp.Emotion =='happiness', 'Emotion'] = 'happy'

    if Emotion_value == 'all':
        df = df_temp
    else:
        df=df_temp.loc[df_temp.Emotion==Emotion_value]

    del df_temp

    vect = CountVectorizer(stop_words='english')
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

@app.callback(
    Output('Pie_chart','figure'), 
    Input('DataSet_dropdown', 'value'))
def display_Pie_Chart_page1(dataset):
    df = pd.read_csv('./Datas/'+dataset)
    if dataset == 'text_emotion.csv':
        df.columns = ["Tweet_id", "Emotion", "Author", "Text"]
        df = df.drop(columns='Tweet_id')
        df = df.drop(columns='Author')
        df.loc[df.Emotion =='happiness', 'Emotion'] = 'happy'
    #### Pie Chart ###
    fig_Pie_chart = go.Figure(data=[go.Pie(labels=df.Emotion.unique(),
                                values=df.groupby('Emotion').Text.nunique(), 
                                textinfo='label+percent',
                                )],
                    layout ={
                    #'title':'Emotions Répartition',
                    'paper_bgcolor':'rgb(22,26,40)',
                    'plot_bgcolor':'rgb(22,26,40)',
                    'font_color':'white',
                    'legend' : {
                        'orientation' : "h",
                        'yanchor':"top",
                        'y':2,
                        'xanchor':"left",
                        'x':-0.2
                        }
                })   
    return fig_Pie_chart

@app.callback(
    Output('page1_table','children'), 
    Input('Emotion_radio', 'value'),
    Input('DataSet_dropdown', 'value'))
def display_table_page1(Emotion_value, dataset):
    df_temp = pd.read_csv('./Datas/'+dataset)
    if dataset == 'text_emotion.csv':
        df_temp.columns = ["Tweet_id", "Emotion", "Author", "Text"]
        df_temp = df_temp.drop(columns='Tweet_id')
        df_temp = df_temp.drop(columns='Author')
        df_temp.loc[df_temp.Emotion =='happiness', 'Emotion'] = 'happy'
        df_temp = df_temp.reindex(columns=['Text','Emotion'])

    if Emotion_value == 'all':
        df = df_temp
    else:
        df=df_temp.loc[df_temp.Emotion==Emotion_value]
        
    del df_temp

    table = dash_table.DataTable(
        id='app-1-table',
        export_format='csv',
        export_headers='display',
        columns=[{'id': c, 'name': c} for c in df.columns],
        data= df.to_dict('records'),
        style_as_list_view=True,
        fixed_rows={'headers': True},
        style_table={
            'overflowX': 'auto',
            'overflowY': 'auto',
            'maxHeight':'400px',
            'minWidth':'70vw',
            'maxWidth':'1600px'
            },

        style_cell_conditional=[
            #{'if': {'column_id': 'Emotion'},'width': '15%'},
            {'if': {'column_id': 'Text'},'width': '58vw'},
            {
            'height': 'auto',
            ## all three widths are needed
            'minWidth': '50px', 'width': '70px', 'maxWidth': '300px',
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



#############################################################################
################################# PAGE 2 ####################################
#############################################################################
@app.callback(
    Output('corr_matrix','figure'), 
    Input('solver_radio', 'value'))
def display_corr_matrix_page2(Model_value):

    y= pipes[Model_value].predict(corpusk)
    cm = metrics.confusion_matrix(targetsk, y)
    fig1 = go.Figure(data=[go.Heatmap(
                    z=cm,
                    x=emot,
                    y=emot,
                    colorscale="Viridis",
                    hovertemplate = "<b>Predicted :</b> %{x} <br>"+"<b>Real :</b> %{y} <br>"
                                   + "<b>Count :</b> %{z} <br>"
                                   + "<extra></extra>")],
                    layout ={
                    'title':'Model : {}'.format(pipes[Model_value].steps[2][0]),
                    'xaxis_title_text': 'Predicted label',
                    'yaxis_title_text': 'Actual label',
                    'paper_bgcolor':'rgb(22,26,40)',
                    'plot_bgcolor':'rgb(22,26,40)',
                    'font_color':'white'
                })
    return fig1
                


@app.callback(
    Output("tab_content", "children"),
    [Input("tabs_fig", "active_tab"), Input('solver_radio', 'value')])
def render_tab_content(active_tab, Model_value):

    model = pipes[Model_value]
    y_scores = model.predict_proba(corpusk)
    y_onehot = pd.get_dummies(targetsk, columns=model.classes_)

    if active_tab is not None:
        if active_tab == "ROC":
            fig_ROC = go.Figure()
            fig_ROC.add_shape(
                type='line', line=dict(dash='dash'),
                x0=0, x1=1, y0=0, y1=1
            )

            for i in range(y_scores.shape[1]):
                y_true = y_onehot.iloc[:, i]
                y_score = y_scores[:, i]

                fpr, tpr, _ = roc_curve(y_true, y_score)
                auc_score = roc_auc_score(y_true, y_score)

                name = f"{emot[y_onehot.columns[i]-1]} (AUC={auc_score:.2f})"
                fig_ROC.add_trace(go.Scatter(x=fpr, y=tpr, name=name, mode='lines'))

            fig_ROC.update_layout(
                xaxis_title = 'False Positive Rate',
                yaxis_title = 'True Positive Rate',
                paper_bgcolor = 'rgb(22,26,40)',
                plot_bgcolor = 'rgb(22,26,40)',
                font_color = 'white',
                yaxis=dict(scaleanchor="x", scaleratio=1),
                xaxis=dict(constrain='domain'),
                width=700, height=500,
                legend = {
                    'yanchor':"bottom",
                    'y':0.05,
                    'xanchor':"left",
                    'x':0.45
                    }
                )
            return dcc.Graph(figure=fig_ROC)

        elif active_tab == "prec_recall":
            fig_prec_recall = go.Figure()
            fig_prec_recall.add_shape(
                type='line', line=dict(dash='dash'),
                x0=0, x1=1, y0=1, y1=0
            )

            for i in range(y_scores.shape[1]):
                y_true = y_onehot.iloc[:, i]
                y_score = y_scores[:, i]

                precision, recall, _ = precision_recall_curve(y_true, y_score)
                auc_score = average_precision_score(y_true, y_score)

                name = f"{y_onehot.columns[i]} (AP={auc_score:.2f})"
                fig_prec_recall.add_trace(go.Scatter(x=recall, y=precision, name=name, mode='lines'))

            fig_prec_recall.update_layout(
                xaxis_title='Recall',
                yaxis_title='Precision',
                yaxis=dict(scaleanchor="x", scaleratio=1),
                xaxis=dict(constrain='domain'),
                paper_bgcolor = 'rgb(22,26,40)',
                plot_bgcolor = 'rgb(22,26,40)',
                font_color = 'white',
                width=700, height=500,
                legend = {
                    'yanchor':"bottom",
                    'y':0.05,
                    'xanchor':"left",
                    'x':0.25
                    }
            )
            return dcc.Graph(figure =fig_prec_recall)
    return "No tab selected"