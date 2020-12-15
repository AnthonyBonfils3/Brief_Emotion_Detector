# Dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table

# librairie classique
import numpy as np
import pandas as pd

# plotly
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff

# sklearn
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer

# nltk
#from nltk.corpus import stopwords as sw
#from nltk.corpus import wordnet as wn
#from nltk import wordpunct_tokenize
#from nltk import WordNetLemmatizer
#from nltk import sent_tokenize
#from nltk import pos_tag
#stopwords = sw.words('english')

#from PIL import Image
## librairie word cloud
#from wordcloud import WordCloud

dfk = pd.read_csv("./Datas/Emotion_final.csv")
emot = dfk.Emotion.unique()
corpusk = dfk.Text
targetsk = dfk.Emotion
targetsk = np.array([1 if x == emot[0] else 2 if x==emot[1] else 3 if x==emot[2] else 4 if x==emot[3] else 5 if x==emot[4] else 6 for x in targetsk])


# Histigramme des émotions
fig1 = go.Figure()
fig1 = go.Figure(
    data=[go.Histogram(x=dfk.Emotion, name='words count'), 
                       go.Histogram(x=dfk.Emotion, cumulative_enabled=True, name='cumulative words count')],
    layout ={
        'title':'Emotions Histogram',
        'xaxis_title_text': 'Emotions',
        'yaxis_title_text': 'Count',
        'paper_bgcolor':'rgb(22,26,40)',
        'plot_bgcolor':'rgb(22,26,40)',
        'font_color':'white',
        'legend' : {
            'yanchor':"top",
            'y':0.99,
            'xanchor':"left",
            'x':0.01
        }})
    

#################################################################################################################
################################################# FIGURES #######################################################
#################################################################################################################

#### histogramme des  mots ####
# Vobabulary analysis
vect = CountVectorizer()#stop_words=stopwords
X = vect.fit_transform(corpusk)
words = vect.get_feature_names()

# Compute rank
wsum = np.array(X.sum(0))[0]
ix = wsum.argsort()[::-1]
wrank = wsum[ix] 
labels = [words[i] for i in ix]

# Sub-sample the data to plot. take the 50 first + the rest sample with the given step 
def subsample(x, end, step=400):
    return np.hstack((x[:30], x[30:end:step]))

trace = go.Bar(x = subsample(labels, 30), y = subsample(wrank,30),
               marker = dict(color = 'rgba(255, 174, 255, 0.5)',
               line = dict(color ='rgb(0,0,0)',width =1.5)),
)
layout = go.Layout(title = "Words ordered by rank. The first rank is the most frequent words and the last one is the less present",
                   xaxis_title_text = 'Word rank',
                   yaxis_title_text = 'word frequency',
                   paper_bgcolor = 'rgb(22,26,40)',
                   plot_bgcolor = 'rgb(22,26,40)',
                   font_color='white')
fig2 = go.Figure(data = trace, layout = layout)

#### Pie Chart ###
fig3 = go.Figure(data=[go.Pie(labels=dfk.Emotion.unique(),
                             values=dfk.groupby('Emotion').Text.nunique(), 
                             textinfo='label+percent',
                            )],
                layout ={
                   'title':'Emotions Répartition',
                   'paper_bgcolor':'rgb(22,26,40)',
                   'plot_bgcolor':'rgb(22,26,40)',
                   'font_color':'white'
               })

#### Wordcloud avec un masque ####

x = subsample(labels, end=8000, step=10)

# def plot_word_cloud(text, masque) :
    
#     mask_coloring = np.array(Image.open(str(masque)))
    
#     # Définir le calque du nuage des mots
#     wc = WordCloud(width=600,
#                    height=600,
#                    background_color="white", 
#                    max_words=200,  
#                    mask = mask_coloring, 
#                    max_font_size=90,
#                    collocations = False, 
#                    random_state=42)

#     # Générer et afficher le nuage de mots
#     plt.figure(figsize= (15,20))
#     wc.generate(" ".join(text))
#     plt.imshow(wc,interpolation="bilinear")
#     plt.axis("off")
#     plt.savefig('wourdcloud-emotions.png')


#plot_word_cloud(x, '/Images/oiseau_tweeter.png')


#################################################################################################################
################################################## LAYOUT #######################################################
#################################################################################################################
layoutPage1 = html.Div([
    html.Header([
        html.H1('Emotion detector'),
        html.H2('Datas Analysis'),

    ]),
    html.Aside(

    ),
    html.Tbody(id='main_block',children=[
        html.Div(id='Block_left', children=[
            html.Article(id='left_selector',children=[
                html.H3('Selectors'),
                html.H3('Graphique'),
                html.Div(id='left-cont-fig', children=[
                    ## Fig 1 : Histigramme Emotions
                    dcc.Graph(
                        id='Hist-emotions',
                        figure=fig1
                        ),
                    ])
            ]),
        ]),
        html.Div(id='Block_right', children=[
            html.Section(id='Block_1',children=[
                html.Article(id='Block_1_Article_1', children=[
                    ## Fig 2 : Histigramme Mots
                    dcc.Graph(
                        id='Hist-mots',
                        figure=fig2
                        ),
                ]),
                html.Article(id='Block_1_Article_2', children=[
                    dcc.Graph(
                        # Fig 3 : Pie Chart
                        id='left-cont-fig',
                        figure=fig3
                        ),
                ]),
            ]),
            html.Section(id='Block_2',children=[
                html.Article(id='Block_2_Article_1', children=[
                    html.H3('Datas Table'), 
                    dash_table.DataTable(
                        id='app-1-table',
                        columns=[{'id': c, 'name': c} for c in dfk.columns],
                        data= dfk.to_dict('records'),
                        style_as_list_view=True,
                        fixed_rows={'headers': True},
                        style_table={
                            'overflowX': 'auto',
                            'overflowY': 'auto',
                            'maxHeight':'400px',
                            'maxWidth':'1600px'},
                        #Cell dim + textpos
                        style_cell_conditional=[{
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
                            'color': 'white',
                            # 'elif':{'row_index': 'even'},
                            # 'backgroundColor': '#1e2130',
                            # 'color': 'white',
                            }],
                        style_header={
                            'backgroundColor': 'rgb(50, 50, 50)',
                            'fontWeight': 'bold',
                            'color':'white'},
                        # Tool Tips
                        tooltip_data=[{
                            column: {'value': str(value), 'type': 'markdown'} for column, value in row.items()
                            } for row in dfk.to_dict('rows')],
                        tooltip_duration=None
                    ),  
                ]),
            ]),        
        ]),
    ]),

    html.Footer([
        html.Br(),
        dcc.Link('Go to Home Page', href='/'),
        # html.Br(),
        # html.Button('Go to Home Page', href='/', n_clicks=0),
        html.Br(),
        dcc.Link("Classifications results", href="/Classifications%20Results")     
    ]),
])