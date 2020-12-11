import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output


from app import app
from layoutHomePage import layoutHome
from layoutPage1 import layoutPage1
from layoutPage2 import layoutPage2
import callbacks

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])

@app.callback(Output('page-content', 'children'),
              Input('url', 'pathname'))
def display_page(pathname):
    if pathname == '/' :
         return layoutHome
    elif pathname == '/apps/app1':
        return layoutPage1
    elif pathname == '/apps/app2':
         return layoutPage2
    else:
        return '404'

if __name__ == '__main__':
    app.run_server(debug=True)