import pandas as pd
import requests
import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.graph_objs as go
import plotly.offline as pyo
from scipy.stats import zscore
import dash
import dash_core_components as dcc
import dash_html_components as html
import datetime
from dash.dependencies import Input, Output
from PIL import Image # new import

#Get Glassnode Data
api_key = "2NmhZcZSARowN6nW069LTzhnZ4L"
TRAIN_START_DATE = '2020-01-01'
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
        'LTHMVRV': 'market/mvrv_more_155',
        'hash_rate':'mining/hash_rate_mean',
        'basis': 'derivatives/futures_annualized_basis_3m',
        'mktcap': 'market/marketcap_usd'

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

def get_stables(indicator, asset):
    endpoints = {
        'supply': 'supply/current'
    }

    url = f"https://api.glassnode.com/v1/metrics/{endpoints[indicator]}?api_key={api_key}&s={start_date}&i=24h&a={asset}"
    response = requests.get(url)

    if response.status_code != 200:
        raise ValueError(f"Error fetching data: {response.status_code}")

    data = pd.DataFrame(response.json())
    data = data.rename(columns={'v': f'{indicator}_{asset}'})
    data['date'] = pd.to_datetime(data['t'], unit='s').dt.strftime('%Y-%m-%d %H:%M:%S')
    data['date_merge'] = pd.to_datetime(data['date'])
    data = data.drop(['t'], axis=1)

    return data  # Add this line to return the created DataFrame

def get_inflow(indicator, asset):
    endpoints = {
        'inflow': 'transactions/transfers_volume_to_exchanges_sum'
    }

    url = f"https://api.glassnode.com/v1/metrics/{endpoints[indicator]}?api_key={api_key}&s={start_date}&i=24h&a={asset}"
    response = requests.get(url)

    if response.status_code != 200:
        raise ValueError(f"Error fetching data: {response.status_code}")

    data = pd.DataFrame(response.json())
    data = data.rename(columns={'v': f'{indicator}_{asset}'})
    data['date'] = pd.to_datetime(data['t'], unit='s').dt.strftime('%Y-%m-%d %H:%M:%S')
    data['date_merge'] = pd.to_datetime(data['date'])
    data = data.drop(['t'], axis=1)

    return data  # Add this line to return the created DataFrame

supply_usdc = get_stables('supply', 'USDC')
supply_usdt = get_stables('supply', 'USDT')
supply_busd = get_stables('supply', 'BUSD')
supply_tusd = get_stables('supply', 'TUSD')
supply_dai = get_stables('supply', 'DAI')

# Fetch inflow data for USDC, USDT, BUSD, TUSD, and DAI
inflow_usdc = get_inflow('inflow', 'USDC')
inflow_usdt = get_inflow('inflow', 'USDT')
inflow_busd = get_inflow('inflow', 'BUSD')
inflow_tusd = get_inflow('inflow', 'TUSD')
inflow_dai = get_inflow('inflow', 'DAI')

# Merge the supply and inflow data into a single DataFrame based on the 'date_merge' column
merged_data = pd.merge(supply_usdc, supply_usdt, on='date_merge', how='outer')
merged_data = pd.merge(merged_data, supply_busd, on='date_merge', how='outer')
merged_data = pd.merge(merged_data, supply_tusd, on='date_merge', how='outer')
merged_data = pd.merge(merged_data, supply_dai, on='date_merge', how='outer')
merged_data = pd.merge(merged_data, inflow_usdc, on='date_merge', how='outer')
merged_data = pd.merge(merged_data, inflow_usdt, on='date_merge', how='outer')
merged_data = pd.merge(merged_data, inflow_busd, on='date_merge', how='outer')
merged_data = pd.merge(merged_data, inflow_tusd, on='date_merge', how='outer')
merged_data = pd.merge(merged_data, inflow_dai, on='date_merge', how='outer')

# Calculate the total inflow for all stablecoins
merged_data['total_inflow'] = merged_data[['inflow_USDC', 'inflow_USDT']].sum(axis=1)

# Calculate the total circulating supply for all stablecoins
merged_data['total_supply'] = merged_data[['supply_USDC', 'supply_USDT']].sum(axis=1)

# Calculate the inflow_supply ratio
merged_data['inflow_supply'] = (merged_data['total_inflow'] / merged_data['total_supply']) * 100

# Merge the OHLC data into the merged_data DataFrame based on the 'date_merge' column
# merged_data = pd.merge(merged_data, ohlc_data, on='date_merge', how='outer')
merged_data['date'] = pd.to_datetime(merged_data['date_merge'])
merged_data = merged_data.drop(['date_x', 'date_y'], axis=1)
df = merged_data

merged_data.to_csv('stablecoins.csv')

ohlc = get_glassnode('ohlc')
funding = get_glassnode('funding')
skew1m = get_glassnode('skew1m')
sth = get_glassnode('STHMVRV')
lth = get_glassnode('LTHMVRV')
hash_rate = get_glassnode('hash_rate')
basis = get_glassnode('basis')
mktcap_data = get_glassnode('mktcap')

df = ohlc.merge(funding, on="date_merge", how="left")
df = df.merge(skew1m, on="date_merge",how="left")
df = df.merge(sth, on="date_merge",how="left")
df = df.merge(lth,on="date_merge",how="left")
df = df.merge(hash_rate,on="date_merge",how="left")
df = df.merge(basis,on="date_merge",how="left")
df = df.merge(mktcap_data,on="date_merge",how="left")
df = df.merge(merged_data,on="date_merge",how="left")

df['date'] = pd.to_datetime(df['date_merge'])
df = df.drop(['date_x', 'date_y','date_merge'], axis=1)
df['skew_100'] = df['skew1m']*100
df['skew_chg'] = df['skew_100'].diff(periods=30)
df['sth'] = df['STHMVRV']*100
df['ratio'] = (df['Close']/df['STHMVRV'])/(df['Close']/df['LTHMVRV'])
df['change14'] = df['ratio'].pct_change(periods = 14)*100
df['hash_30'] = df['hash_rate'].rolling(window=30).mean()
df['hash_60'] = df['hash_rate'].rolling(window=60).mean()
df['hashcross'] = (df['hash_30']/df['hash_60'])*100
df['basis_30'] = (df['basis'].diff(periods=30))*100
df['bitcoin_yardstick'] = df['mktcap'] / df['hash_rate']

# Calculate rolling Z-Score
rolling_mean = df['bitcoin_yardstick'].rolling(window=2*365).mean()
rolling_std = df['bitcoin_yardstick'].rolling(window=2*365).std()
df['rolling_zscore'] = (df['bitcoin_yardstick'] - rolling_mean) / rolling_std

df.to_csv('dash.csv')
metric_cleaned = df['funding'].dropna()

# Create a subplot with 2 rows and 2 columns, and add subplot titles for tab 1
fig = make_subplots(
    rows=4, cols=2,
    subplot_titles=["Perps Funding Rate", "Skew 1M 30d Change",
                    "STH Price Cross", "STHLTH Ratio 14d Change",
                    "Hash Ribbons", 'Funding Basis 30d Change',
                    "Stablecoin Inflows USDT & USDC", "Hash Rate Deviation"],
    row_heights=[0.8, 0.8, 0.8, 0.8],
    vertical_spacing=0.1,  # Adjust the vertical spacing between subplots
)

'''
# Create a subplot with 2 rows and 2 columns, and add subplot titles for tab 2
fig2 = make_subplots(
    rows=1, cols=2,
    subplot_titles=["Test 1", "Test 2"],
    row_heights=[0.8],
    vertical_spacing=0.1,  # Adjust the vertical spacing between subplots
)
'''

# Function to convert color to pastel shade
def pastel_color(hex_color):
    r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    return f"rgba({r}, {g}, {b}, 0.5)"

# Chart 1
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

# Chart 2 
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

# Chart 3
for index, row in df.iterrows():
    color = pastel_color('008000') if row['sth'] >= 100 else pastel_color('FF0000')
    trace = go.Scatter(x=[row['date'], row['date']], y=[0, row['sth']], mode='lines', name='',
                       line=dict(color=color), showlegend=False)  # Hide legend entry for Chart 3
    fig.add_trace(trace, row=2, col=1)

# Add static second line at y=98 for Chart 3
static_line_sth = [100] * len(df['date'])
sth_static_line = go.Scatter(x=df['date'], y=static_line_sth, mode='lines', name='',
                             line=dict(color=pastel_color('000080'), dash='dash'), showlegend=False)  # Hide legend entry for Static Line
fig.add_trace(sth_static_line, row=2, col=1)

# Chart 4 
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

# Chart 5 
for index, row in df.iterrows():
    color = pastel_color('008000') if row['hashcross'] >= 100 else pastel_color('FF0000')
    trace = go.Scatter(x=[row['date'], row['date']], y=[0, row['hashcross']], mode='lines', name='',
                       line=dict(color=color), showlegend=False)  # Hide legend entry for Chart 4
    fig.add_trace(trace, row=3, col=1)
    
# Add static second line at y=0 for Chart 5
static_line_chart5 = [100] * len(df['date'])
chart5_static_line = go.Scatter(x=df['date'], y=static_line_chart5, mode='lines', name='',
                                line=dict(color=pastel_color('000080'), dash='dash'), showlegend=False)
fig.add_trace(chart5_static_line, row=3, col=1)

# Chart 6 
for index, row in df.iterrows():
    color = pastel_color('008000') if row['basis_30'] >= 0 else pastel_color('FF0000')
    trace = go.Scatter(x=[row['date'], row['date']], y=[0, row['basis_30']], mode='lines', name='',
                       line=dict(color=color), showlegend=False)  # Hide legend entry for Chart 4
    fig.add_trace(trace, row=3, col=2)
    
# Add static line for Chart 6
static_line_chart6 = [3] * len(df['date'])
chart6_static_line = go.Scatter(x=df['date'], y=static_line_chart6, mode='lines', name='',
                                line=dict(color=pastel_color('000080'), dash='dash'), showlegend=False)
fig.add_trace(chart6_static_line, row=3, col=2)

#Chart 7 
for index, row in df.iterrows():
    color = pastel_color('008000') if row['inflow_supply'] >= 0 else pastel_color('FF0000')
    trace = go.Scatter(x=[row['date'], row['date']], y=[0, row['inflow_supply']], mode='lines', name='',
                       line=dict(color=color), showlegend=False)  # Hide legend entry for Chart 7
    fig.add_trace(trace, row=4, col=1)

# Add static line for Chart 7
static_line_chart7 = [3] * len(df['date'])
chart7_static_line = go.Scatter(x=df['date'], y=static_line_chart7, mode='lines', name='',
                                line=dict(color=pastel_color('000080'), dash='dash'), showlegend=False)
fig.add_trace(chart7_static_line, row=4, col=1)

#Chart8
for index, row in df.iterrows():
    if row['rolling_zscore'] >= 4:
        color = pastel_color('FF0000')  # Red for over 4
    elif row['rolling_zscore'] >= 3:
        color = pastel_color('FFA500')  # Orange for above 3 and below 4
    elif row['rolling_zscore'] < -1.5:
        color = pastel_color('0000FF')  # Blue for below -1.5
    elif row['rolling_zscore'] < -1:
        color = pastel_color('008000')  # Green for below -1 and above -1.5
    else:
        color = 'rgba(169, 169, 169, 0.5)'  # Darker gray for default
        
    trace = go.Scatter(x=[row['date'], row['date']], y=[0, row['rolling_zscore']], mode='lines', name='',
                       line=dict(color=color), showlegend=False)
    fig.add_trace(trace, row=4, col=2)

# Add static line for Chart 8
static_line_chart8 = [-1] * len(df['date'])
chart8_static_line = go.Scatter(x=df['date'], y=static_line_chart8, mode='lines', name='',
                                line=dict(color=pastel_color('000080'), dash='dash'), showlegend=False)
fig.add_trace(chart8_static_line, row=4, col=2)

'''
# Create Chart And Percentile Metrics
df_chain = pd.read_csv('metrics.csv')
df_chain = df_chain[(df_chain['t'] > "2016-01-01")]
df_chain = df_chain.sort_values(by="t")

df_chain['rhodl_percentile'] = df_chain.rhodl_ratio.rank(pct = True)
df_chain['mvrv_z_percentile'] = df_chain.mvrv_z_score.rank(pct = True)
df_chain['reserve_risk_percentile'] = df_chain.reserve_risk.rank(pct = True)
df_chain['puell_multiple_percentile'] = df_chain.puell_multiple.rank(pct = True)
df_chain['nupl_percentile'] = df_chain.net_unrealized_profit_loss.rank(pct = True)
df_chain['profit_percentile'] = (df_chain.profit_relative.rolling(window=30).mean()).rank(pct = True)
df_chain['30_day_realized_price'] = (df_chain.price_realized_usd.pct_change(periods=30)).rank(pct = True)
df_chain['30_day_price'] = (df_chain.price_usd_close.pct_change(periods=30)).rank(pct = True)
df_chain['mayer_percentile'] = (df_chain.price_usd_close/df_chain.price_usd_close.rolling(window=200).mean()).rank(pct = True)
df_chain['30_day_hash_rate'] = (df_chain.hash_rate_mean.rolling(window=7).mean()).rank(pct = True)
df_chain['rhodl_z'] = zscore(df_chain['rhodl_ratio'])
df_chain['reserve_risk_z'] = zscore(df_chain['reserve_risk'])
df_chain['sthlth_ratio'] = (df_chain['price_usd_close']/df_chain['mvrv_less_155'])/(df_chain['price_usd_close']/df_chain['mvrv_more_155'])
df_chain['sthlth_ratio_percentile'] = df_chain['sthlth_ratio'].rank(pct=True)

#Create Index
df_chain['agg'] = df_chain['rhodl_percentile'] + df_chain['mvrv_z_percentile'] + df_chain['reserve_risk_percentile'] + df_chain['puell_multiple_percentile'] + df_chain['nupl_percentile'] + df_chain['profit_percentile'] + df_chain['30_day_realized_price'] + df_chain['mayer_percentile'] + df_chain['sthlth_ratio_percentile']    
df_chain['agg'] = df_chain['agg']/9
    
#Create Seperate df_chain for heatmap chart 
df_chain2 = df_chain[['t', 
          'rhodl_percentile', 
          'mvrv_z_percentile', 
          'reserve_risk_percentile',
          'puell_multiple_percentile', 
          'nupl_percentile', 
          'profit_percentile',
          '30_day_realized_price',
          'mayer_percentile',
          'sthlth_ratio_percentile',
          'agg'
         ]]

# Aggregate percentile chart 
colors = ['red', 'green', 'blue', 'yellow', 'orange']

# Create the heatmap trace
trace = go.Heatmap(x=df_chain2['t'], y=df_chain2.columns[1:], z=df_chain2.iloc[:, 1:].values.T, colorscale='RdYlGn_r', dy=10)

# Create subplots with increased height
fig_heat = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, row_heights=[0.4, 0.4], subplot_titles=("On-Chain Metrics Heatmap", "BTC Price Weighted"))

# Add heatmap trace to the first subplot
fig_heat.add_trace(trace, row=1, col=1)

# Add scatter chart trace to the second subplot (with log y-axis)
fig_heat.add_trace(go.Scatter(x=df_chain['t'], y=df_chain['price_usd_close'], mode='markers', marker=dict(color=df_chain['agg'], colorscale='RdYlGn_r')), row=2, col=1)

# Update layout of the subplots
fig_heat.update_layout(
    title='BTC On-Chain Metrics and Price',
    #xaxis_title='Date',
    yaxis_title='Percentile',
    yaxis2_title='Price (USD, Log Scale)',
    plot_bgcolor='white',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(family="Arial", size=12, color="black"),  # Set text color to black
    title_font=dict(family="Arial", size=18, color="black"),  # Set title text color to black
    yaxis2_type='log'  # Set y-axis of the second subplot to log scale
)
'''
# Update layout and axis labels
fig.update_layout(
    title_text="UTXO Dashboard",
    height=1200,  # Increased height to create space for annotations
    plot_bgcolor='rgb(240, 240, 240)',
    font_family='Arial',
    font_size=20,
)

'''
# Update the height of the subplots
fig_heat.update_layout(height=1000)  # Adjust the height of the entire subplot
'''
# Calculate the date 6 months ago from the current date
six_months_ago = datetime.datetime.now() - datetime.timedelta(days=6*30)

# Update X-axis range for all subplots in the 1st column to start from 6 months ago
for i in range(1, 5):
    fig.update_xaxes(range=[six_months_ago, df['date'].max()], row=i, col=1)

# Update X-axis range for all subplots in the 2nd column to start from 6 months ago
for i in range(1, 5):
    fig.update_xaxes(range=[six_months_ago, df['date'].max()], row=i, col=2)

# Update Y Axis Across Charts
fig.update_yaxes(range=[-0.0005, 0.0005], row=1, col=1)

fig.update_yaxes(range=[50, 150], row=2, col=1)
fig.update_yaxes(range=[-10, 20], row=2, col=2)

fig.update_yaxes(range=[90, 110], row=3, col=1)
fig.update_yaxes(range=[-10, 10], row=3, col=2)

fig.update_yaxes(range=[0, 10], row=4, col=1)
fig.update_yaxes(range=[-2, 3], row=4, col=2)

# Add annotations at the bottom of the charts
note_text = "Green: Positive values | Red: Negative values | Blue lines: Entry triggers"

fig.add_annotation(xref='paper', yref='paper',text=note_text, y=-0.08,
                   showarrow=False, font=dict(size=12), align='center')


fig.update_xaxes(tickfont=dict(size=12))
fig.update_yaxes(tickfont=dict(size=12))

# Show the dashboard
fig.show()

# Create the Dash app
app = dash.Dash(__name__)
server = app.server

# Define the layout of the app with tabs
app.layout = html.Div([
    dcc.Tabs(
        id='tabs',
        value='tab-1',
        children=[
            dcc.Tab(label='Strategy Tracking', value='tab-1'),
            dcc.Tab(label='On-Chain Heatmap', value='tab-2'),
        ],
        colors={
            "border": "white",
            "primary": "dodgerblue",
            "background": "aliceblue"
        }
    ),
    html.Div(id='tabs-content', style={'padding': '20px'})
])


# Define callback to update the content based on selected tab
@app.callback(Output('tabs-content', 'children'), [Input('tabs', 'value')])
def render_content(tab):
    if tab == 'tab-1':
        # Return the figure for Tab 1
        return dcc.Graph(figure=fig)
    elif tab == 'tab-2':
        return dcc.Graph(figure=fig)

if __name__ == '__main__':
    app.run_server(debug=False)
