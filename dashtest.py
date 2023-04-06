import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from PIL import Image # new import

# Create Chart And Percentile Metrics
df = pd.read_csv('metrics.csv')
df['test'] = df.rhodl_ratio.rank(pct = True)
df['test2'] = df.mvrv_z_score.rank(pct = True)
df['200davg'] = df.price_usd_close.rolling(window=200).mean()

fig = px.scatter(df, x='t', y='price_usd_close', color='test', 
                 log_y=True, title="BTC Price Weighted By RHODL Percentile",
                 width=800, height=400)

fig2 = px.scatter(df, x='t', y='price_usd_close', color='test2', 
                 log_y=True, title="BTC Price Weighted By MVRV Z Percentile",
                 width=800, height=400)

img = Image.open('LOGOsmall.png') # image path

layout = go.Layout(
    title="Data",
    plot_bgcolor="#FFF",  # Sets background color to white
    xaxis=dict(
        title="time",
        linecolor="#BCCCDC",  # Sets color of X-axis line
        showgrid=False  # Removes X-axis grid lines
    ),
    yaxis=dict(
        title="price",  
        linecolor="#BCCCDC",  # Sets color of Y-axis line
        showgrid=False,  # Removes Y-axis grid lines    
    )
)

fig3 = go.Figure(
    data=go.Scatter(x=df['t'],y=df['price_usd_close'], name = 'price'),
    layout=layout
) 

# defined to source
fig3.add_layout_image(
    dict(
        source=img,
        xref="paper", yref="paper",
        x=0.5, y=0.24,
        sizex=0.5, sizey=0.6,
        xanchor="center",
        yanchor="bottom",
        opacity=0.25
    )
)
fig3.update_yaxes(type='log')
fig3.add_trace(go.Scatter(x=df['t'], y=df['200davg'], name = '200d avg'))
fig3.add_trace(go.Scatter(x=df['t'], y=df['price_realized_usd'], name = 'Realized Price'))
fig3.update_layout(width=1200, height=600)
title_text = "BTC Price Versus Key Price Levels"
fig3.update_layout(title=title_text)

fig3.update_layout(
    xaxis=dict(
        rangeselector=dict(
            buttons=list([
                dict(count=1,
                     label="1m",
                     step="month",
                     stepmode="backward"),
                dict(count=6,
                     label="6m",
                     step="month",
                     stepmode="backward"),
                dict(count=1,
                     label="YTD",
                     step="year",
                     stepmode="todate"),
                dict(count=1,
                     label="1y",
                     step="year",
                     stepmode="backward"),
                dict(step="all")
            ])
        ),
        rangeslider=dict(
            visible=True
        ),
        type="date"
    )
)

#Dash Example
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
 
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server
 
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
            figure=fig3
        ),  
    ]),
])

if __name__ == '__main__':
    app.run_server(debug=False)

