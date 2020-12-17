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

list_emot = list(emot)
list_emot.append('all')


# Histigramme des émotions
fig1 = go.Figure()
fig1 = go.Figure(
    data=[go.Histogram(x=dfk.Emotion, name='words count'), 
                       go.Histogram(x=dfk.Emotion, cumulative_enabled=True, name='cumulative words count')],
    layout ={
        #'title':'Emotions Histogram',
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
            }
        })
    

#################################################################################################################
################################################# FIGURES #######################################################
#################################################################################################################

#### Pie Chart ###
fig3 = go.Figure(data=[go.Pie(labels=dfk.Emotion.unique(),
                             values=dfk.groupby('Emotion').Text.nunique(), 
                             textinfo='label+percent',
                            )],
                layout ={
                   #'title':'Emotions Répartition',
                   'paper_bgcolor':'rgb(22,26,40)',
                   'plot_bgcolor':'rgb(22,26,40)',
                   'font_color':'white'
               })

#### Wordcloud avec un masque ####

# x = subsample(labels, end=8000, step=10)

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
                        {'label': 'First one (approx 20k entries): Emotion_final.csv', 'value': 'Emotion_final.csv'},
                        {'label': 'Second one (approx 40k entries): text_emotion.csv', 'value': 'text_emotion.csv'},
                    ],
                    optionHeight= 60,
                    value='Emotion_final.csv',
                    clearable=False,
                ),

                ## Emotion selector
                html.H3('Select an emotion'),
                dcc.RadioItems(
                   id='Emotion_radio',
                   options=[{'label': k, 'value': k} for k in list_emot],
                   value = 'all'
                   ),
                html.H3('Emotions histogram'),

                ## Fig 1 : Histigramme Emotions
                dcc.Graph(
                    id='Hist_emotions',
                    figure=fig1
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
                        )
                ]),

                html.Article(id='Block_1_Article_2', children=[
                    html.H3('Emotions Répartition'),
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
                    # Tableau des données
                    html.Div(id='page1_table')
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