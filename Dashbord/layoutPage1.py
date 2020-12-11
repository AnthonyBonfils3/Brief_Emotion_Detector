import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import dash_table

df = pd.read_csv("./Datas/Emotion_final.csv")

layoutPage1 = html.Div([
    html.H3('Table des données'),
    dash_table.DataTable(
    id='app-1-dropdown',
    columns=[{'id': c, 'name': c} for c in df.columns],
    data= df.to_dict('records'),
    style_as_list_view=True,
    fixed_rows={'headers': True},
    style_table={'overflowX': 'auto','overflowY': 'auto','maxHeight':'800px','maxWidth':'1600px'},
    #Cell dim + textpos
    style_cell_conditional=[{'height': 'auto',
        # all three widths are needed
        'minWidth': '180px', 'width': '180px', 'maxWidth': '180px',
        'whiteSpace': 'normal','textAlign':'center'}
    ],
    #Line strip
    style_data_conditional=[
        {
            'if': {'row_index': 'odd'},
            'backgroundColor': 'rgb(248, 248, 248)'
        }
    ],
    style_header={
        'backgroundColor': 'rgb(50, 50, 50)',
        'fontWeight': 'bold',
        'color':'white'},
)
])