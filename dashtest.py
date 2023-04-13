import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objects as go
import plotly.express as px
import plotly.offline as pyo
import pandas as pd
from PIL import Image # new import

# Create Chart And Percentile Metrics
df = pd.read_csv('metrics.csv')
df = df[(df['t'] > "2012-01-01")]

df['test'] = df.rhodl_ratio.rank(pct = True)
df['test2'] = df.mvrv_z_score.rank(pct = True)
df['200davg'] = df.price_usd_close.rolling(window=200).mean()
df['200wavg'] = df.price_usd_close.rolling(window=1400).mean()
df['rhodl_percentile'] = df.rhodl_ratio.rank(pct = True)
df['mvrv_z_percentile'] = df.mvrv_z_score.rank(pct = True)
df['reserve_risk_percentile'] = df.reserve_risk.rank(pct = True)
df['puell_multiple_percentile'] = df.puell_multiple.rank(pct = True)
df['nupl_percentile'] = df.net_unrealized_profit_loss.rank(pct = True)

img = Image.open('LOGOsmall.png') # image path

df2 = df[['t', 'rhodl_percentile', 'mvrv_z_percentile', 'reserve_risk_percentile','puell_multiple_percentile', 'nupl_percentile']]

# Define the heatmap trace
trace = go.Heatmap(x=df2['t'], y=df2.columns[1:], z=df2.iloc[:, 1:].values.T, colorscale='RdYlGn', dy=10, reversescale=True)

# Define the layout
heatlayout = go.Layout(title='BTC On-Chain Cycle Metrics Heatmap by Percentile')

pricelayout = go.Layout(
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
    layout=pricelayout
) 

# Create the figure object and plot it
fig = go.Figure(data=[trace], layout=heatlayout)
##fig.layout.height = 800
#pyo.plot(fig, filename='heatmap.html')

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
fig3.add_trace(go.Scatter(x=df['t'], y=df['price_realized_usd'], name = 'Realized Price'))
fig3.add_trace(go.Scatter(x=df['t'], y=df['200davg'], name = '200d avg'))
fig3.add_trace(go.Scatter(x=df['t'], y=df['200wavg'], name = '200w avg'))
title_text = "BTC Price Versus Key Price Levels"
fig3.update_layout(title=title_text)

fig3.update_yaxes(fixedrange=False)

#Dash Example
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
 
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server
 
colors = {
    'background': '#F0F8FF',
    'text': '#00008B'
}

app.layout = html.Div(className='row', children=[
    html.H1("BM PRO Dashboard MVP"),
    html.Div(children=[
        dcc.Graph(id="graph1", figure=fig3, style={'display': 'inline-block'}),
        dcc.Graph(id="graph2", figure=fig, style={'display': 'inline-block'})
        
     ]),
    html.Div(children=[
        dcc.Graph(id="graph3", figure=fig3, style={'display': 'inline-block'}),
        dcc.Graph(id="graph4", figure=fig, style={'display': 'inline-block'})
    ])
])

if __name__ == '__main__':
    app.run_server(debug=False)
    
