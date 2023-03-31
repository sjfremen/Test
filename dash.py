import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
import pandas as pd


# Create Chart And Percentile Metrics
df = pd.read_csv('metrics.csv')
df['test'] = df.rhodl_ratio.rank(pct = True)
df['test2'] = df.mvrv_z_score.rank(pct = True)

fig = px.scatter(df, x='t', y='price_usd_close', color='test', 
                 log_y=True, title="BTC Price Weighted By RHODL Percentile",
                 width=800, height=400)

fig2 = px.scatter(df, x='t', y='price_usd_close', color='test2', 
                 log_y=True, title="BTC Price Weighted By MVRV Z Percentile",
                 width=800, height=400)
##fig.show()

#Dash Example
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
 
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
 
colors = {
    'background': '#F0F8FF',
    'text': '#00008B'
}

app.layout = html.Div(children=[
    # All elements from the top of the page
    html.Div([
        html.H1(children='Hello Dash'),

        html.Div(children='''
            Dash: A web application framework for Python.
        '''),

        dcc.Graph(
            id='graph1',
            figure=fig
        ),  
    ]),
    # New Div for all elements in the new 'row' of the page
    html.Div([
        html.H1(children='Hello Dash'),

        html.Div(children='''
            Dash: A web application framework for Python.
        '''),

        dcc.Graph(
            id='graph2',
            figure=fig2
        ),  
    ]),
])

if __name__ == '__main__':
    app.run_server(debug=False)

