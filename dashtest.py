#Get Glassnode Data
import pandas as pd
import requests
import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
import dash_core_components as dcc
import dash_html_components as html

api_key = "2NmhZcZSARowN6nW069LTzhnZ4L"
TRAIN_START_DATE = '2023-01-01'
dt = datetime.datetime.strptime(TRAIN_START_DATE, '%Y-%m-%d')
start_date = int(dt.timestamp())

def get_glassnode(indicator):
  endpoints = {
        'mvrv': 'market/mvrv_z_score',
        'ohlc': 'market/price_usd_ohlc',
        'funding': 'derivatives/futures_funding_rate_perpetual',
        'oi': 'derivatives/futures_open_interest_sum',
        'skew1w': 'derivatives/options_25delta_skew_1_week',
        'skew1m': 'derivatives/options_25delta_skew_1_month',
        'STHMVRV': 'market/mvrv_less_155',
        'LTHMVRV': 'market/mvrv_more_155'

  }
  
  url_mvrv = f"https://api.glassnode.com/v1/metrics/{endpoints[indicator]}?api_key={api_key}&s={start_date}&i=24h&a=BTC"
  response_mvrv = requests.get(url_mvrv)

  if response_mvrv.status_code != 200:
      raise ValueError(f"Error fetching MVRV data: {response_mvrv.status_code}")
  

  if indicator == 'ohlc':
    data_dict = response_mvrv.json()
    mvrv_data = pd.json_normalize(data_dict, sep='_')
    mvrv_data.rename(columns={'o_o': 'Open', 'o_h': 'High',
                     'o_l': 'Low', 'o_c': 'Close', 'o_v': 'Volume'}, inplace=True)


  else:      
    mvrv_data = pd.DataFrame(response_mvrv.json())
    mvrv_data = mvrv_data.rename(columns={'v': indicator})

  mvrv_data['date'] = pd.to_datetime(mvrv_data['t'], unit='s').dt.strftime('%Y-%m-%d %H:%M:%S')
  mvrv_data['date_merge'] = pd.to_datetime(mvrv_data['date'])
  mvrv_data = mvrv_data.drop(['t'], axis=1)

  # mvrv_data.head()
  return mvrv_data


ohlc = get_glassnode('ohlc')
funding = get_glassnode('funding')
skew1m = get_glassnode('skew1m')
sth = get_glassnode('STHMVRV')
lth = get_glassnode('LTHMVRV')

df = ohlc.merge(funding, on="date_merge", how="left")
df = df.merge(skew1m, on="date_merge",how="left")
df = df.merge(sth, on="date_merge",how="left")
df = df.merge(lth,on="date_merge",how="left")
df['date'] = pd.to_datetime(df['date_merge'])
df = df.drop(['date_x', 'date_y','date_merge'], axis=1)
df['skew_100'] = df['skew1m']*100
df['skew_chg'] = df['skew_100'].diff(periods=30)
df['sth'] = df['STHMVRV']*100
df['ratio'] = (df['Close']/df['STHMVRV'])/(df['Close']/df['LTHMVRV'])
df['change14'] = df['ratio'].pct_change(periods = 14)*100

df.to_csv('dash.csv')
metric_cleaned = df['funding'].dropna()

import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Create a subplot with 2 rows and 2 columns, and add subplot titles
fig = make_subplots(rows=2, cols=2, subplot_titles=["Perps Funding Rate", "Skew 1M 30d Change",
                                                    "STH Price Cross", "STHLTH Ratio 14d Change"],
                    row_heights=[0.8, 0.8])

# Function to convert color to pastel shade
def pastel_color(hex_color):
    r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    return f"rgba({r}, {g}, {b}, 0.5)"

# Chart 1 (Top-left) - Replicating Chart 1 from the previous code
for index, row in df.iterrows():
    color = pastel_color('008000') if row['funding'] >= 0 else pastel_color('FF0000')
    trace = go.Scatter(x=[row['date'], row['date']], y=[0, row['funding']], mode='lines', name='',
                       line=dict(color=color), showlegend=False)  # Hide legend entry for Chart 1
    fig.add_trace(trace, row=1, col=1)

# Add static second line at y=-0.000052 - Replicating static line for Chart 1
static_line_y = [-0.000052] * len(df['date'])
trace2_static_line = go.Scatter(x=df['date'], y=static_line_y, mode='lines', name='',
                               line=dict(color=pastel_color('000080'), dash='dash'), showlegend=False)  # Hide legend entry for Static Line
fig.add_trace(trace2_static_line, row=1, col=1)

# Chart 2 (Top-right) - Original Chart 2
for index, row in df.iterrows():
    color = pastel_color('008000') if row['skew_chg'] >= 0 else pastel_color('FF0000')
    trace = go.Scatter(x=[row['date'], row['date']], y=[0, row['skew_chg']], mode='lines', name='',
                       line=dict(color=color), showlegend=False)  # Hide legend entry for Chart 2
    fig.add_trace(trace, row=1, col=2)

# Add static second line at y=12 for Chart 2
static_line_skew = [12] * len(df['date'])
skew_static_line = go.Scatter(x=df['date'], y=static_line_skew, mode='lines', name='',
                              line=dict(color=pastel_color('000080'), dash='dash'), showlegend=False)  # Hide legend entry for Static Line
fig.add_trace(skew_static_line, row=1, col=2)

# Chart 3 (Bottom-left) - Reflecting variable 'sth' in Chart 3
for index, row in df.iterrows():
    color = pastel_color('008000') if row['sth'] >= 98 else pastel_color('FF0000')
    trace = go.Scatter(x=[row['date'], row['date']], y=[0, row['sth']], mode='lines', name='',
                       line=dict(color=color), showlegend=False)  # Hide legend entry for Chart 3
    fig.add_trace(trace, row=2, col=1)

# Add static second line at y=98 for Chart 3
static_line_sth = [98] * len(df['date'])
sth_static_line = go.Scatter(x=df['date'], y=static_line_sth, mode='lines', name='',
                             line=dict(color=pastel_color('000080'), dash='dash'), showlegend=False)  # Hide legend entry for Static Line
fig.add_trace(sth_static_line, row=2, col=1)

# Chart 4 (Bottom-right) - Reflecting variable 'change14' in Chart 4
for index, row in df.iterrows():
    color = pastel_color('008000') if row['change14'] >= 0 else pastel_color('FF0000')
    trace = go.Scatter(x=[row['date'], row['date']], y=[0, row['change14']], mode='lines', name='',
                       line=dict(color=color), showlegend=False)  # Hide legend entry for Chart 4
    fig.add_trace(trace, row=2, col=2)

# Add static second line at y=0 for Chart 4
static_line_change14 = [1] * len(df['date'])
change14_static_line = go.Scatter(x=df['date'], y=static_line_change14, mode='lines', name='',
                                  line=dict(color=pastel_color('000080'), dash='dash'), showlegend=False)  # Hide legend entry for Static Line
fig.add_trace(change14_static_line, row=2, col=2)

# Update layout and axis labels
fig.update_layout(title_text="UTXO Strategy Dashboard", height=800, plot_bgcolor='rgb(240, 240, 240)',
                  font_family='Arial', font_size=14)

# Add annotations at the bottom of the charts
note_text = "Blue dotted lines are entry triggers"
color_note_text = "Green: Positive values | Red: Negative values"
fig.add_annotation(xref='paper', yref='paper', x=0.5, y=-0.10, text=note_text,
                   showarrow=False, font=dict(size=12), align='center')
fig.add_annotation(xref='paper', yref='paper', x=0.5, y=-0.13, text=color_note_text,
                   showarrow=False, font=dict(size=12), align='center')

# Show the dashboard
fig.show()


# Create the Dash app
app = dash.Dash(__name__)
server = app.server

# Define the layout of the app
app.layout = html.Div([
    dcc.Graph(id='output-graph', figure=fig)  # The figure is initially set to the 'fig' you defined earlier
])

@app.callback(
    dash.dependencies.Output('output-graph', 'figure'),
)
def update_graph():
    return fig

if __name__ == '__main__':
    app.run_server(debug=False)
