import dash
from dash import dcc, html, Input, Output, State
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.preprocessing import StandardScaler
import webbrowser
import threading
import time
import numpy as np
from functools import lru_cache
import dash_bootstrap_components as dbc

# Load data with cache
@lru_cache(maxsize=10)
def load_data():
    df = pd.read_csv(r'C:/Users/abhis/OneDrive/Desktop/Energy data/synthetic_energy_data.csv.csv')
    if 'appliance' not in df.columns:
        df['appliance'] = np.random.choice(['Heater', 'AC', 'Washer', 'Fridge', 'Lights'], size=len(df))
    return df

# Model training
df_initial = load_data()
X = df_initial.drop(['consumption', 'appliance'], axis=1)
y = df_initial['consumption']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_scaled, y)

# App setup
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True
)
app.title = "Energy Dashboard"

# Sidebar
sidebar = html.Div([
    html.H2("⚡ Dashboard", className="display-5"),
    html.Hr(),
    dbc.Nav([
        dbc.NavLink("Home", href="/", active="exact"),
        dbc.NavLink("Predictions", href="/predict", active="exact"),
        dbc.NavLink("Cost Estimation", href="/cost", active="exact"),
        dbc.NavLink("Smart Scheduler", href="/smart", active="exact"),
        dbc.NavLink("Anomalies & Tips", href="/anomalies", active="exact")
    ], vertical=True, pills=True),
], style={"position": "fixed", "top": 0, "left": 0, "bottom": 0, "width": "16rem", "padding": "2rem 1rem", "backgroundColor": "#f8f9fa"})

# Main content
content = html.Div(id="page-content", style={"marginLeft": "18rem", "marginRight": "2rem", "padding": "2rem 1rem"})

app.layout = html.Div([dcc.Location(id="url"), sidebar, content])

# ---------------- HOME ----------------
home_layout = html.Div([
    html.H3("📊 Energy Overview", className='mb-4'),
    dbc.Row([
        dbc.Col(dbc.Card([dbc.CardBody([html.H5("Total Units Today"), html.H2(id='total-units')])]), width=4),
        dbc.Col(dbc.Card([dbc.CardBody([html.H5("Estimated Cost (₹)"), html.H2(id='estimated-cost')])]), width=4),
        dbc.Col(dbc.Card([dbc.CardBody([html.H5("Peak Hour"), html.H2(id='peak-hour')])]), width=4)
    ], className="mb-4"),
    dbc.Row([
        dbc.Col([html.Label("Day"), dcc.Input(id='filter-day', type='number')], width=2),
        dbc.Col([html.Label("Month"), dcc.Input(id='filter-month', type='number')], width=2),
        dbc.Col([html.Label("Year (ignore if static data)"), dcc.Input(id='filter-year', type='number')], width=2)
    ]),
    html.Br(),
    dcc.Dropdown(id='x-axis-choice',
                 options=[{'label': col, 'value': col} for col in df_initial.columns if col not in ['consumption', 'appliance']],
                 value='hour_of_day', style={'width': '60%'}),
    dcc.Graph(id='main-bar-graph'),
    html.Hr(),
    dcc.Graph(id='line-hourly-consumption'),
    html.Hr(),
    html.H4("📌 Appliance-wise Consumption Share"),
    dcc.Graph(id='appliance-pie-chart'),
    html.Hr(),
    html.H4("📊 Hourly Consumption Distribution"),
    dcc.Graph(id='box-hourly-consumption'),
    html.Hr(),
    html.H4("🌡️ Heatmap: Day vs Hour Consumption"),
    dcc.Graph(id='heatmap-consumption'),
    dcc.Interval(id='interval-dashboard', interval=60000, n_intervals=0)
])

# ---------------- PREDICT ----------------
predict_layout = html.Div([
    html.H3("🔮 Predict Hourly & Daily Consumption"),
    dbc.Row([
        dbc.Col([html.Label("3_levels"), dcc.Input(id='3lv', type='number', value=2)], width=3),
        dbc.Col([html.Label("5_levels"), dcc.Input(id='5lv', type='number', value=2)], width=3),
        dbc.Col([html.Label("7_levels"), dcc.Input(id='7lv', type='number', value=4)], width=3),
        dbc.Col([html.Label("Temperature"), dcc.Input(id='temp', type='number', value=-5)], width=3)
    ]),
    dbc.Row([
        dbc.Col([html.Label("Hour of Day"), dcc.Input(id='hour', type='number', value=8)], width=3),
        dbc.Col([html.Label("Day of Week"), dcc.Input(id='dow', type='number', value=1)], width=3),
        dbc.Col([html.Label("Day of Month"), dcc.Input(id='dom', type='number', value=8)], width=3),
        dbc.Col([html.Label("Month of Year"), dcc.Input(id='moy', type='number', value=1)], width=3)
    ]),
    html.Div(id='prediction-output', className='mt-3'),
    html.Button("📅 Predict Next Day Consumption", id='predict-day-btn', n_clicks=0),
    html.Div(id='next-day-output', className='mt-3')
])

# ---------------- COST ----------------
cost_layout = html.Div([
    html.H3("💸 Cost Estimation"),
    html.Label("Enter cost per unit (₹):"),
    dcc.Input(id='unit-cost', type='number', value=8),
    html.Div(id='cost-estimate-output', className='mt-2')
])

# ---------------- SMART ----------------
smart_layout = html.Div([
    html.H3("🏭 Smart Scheduler"),
    dcc.Graph(id='smart-schedule-graph'),
    html.Div(id='smart-schedule-tips', className='mt-3'),
    dcc.Interval(id='interval-smart', interval=60000, n_intervals=0)
])

# ---------------- ANOMALIES ----------------
anomaly_layout = html.Div([
    html.H3("🛑 Anomalies & Tips"),
    dcc.Graph(id='anomaly-graph'),
    html.Div(id='recommendations-output', className='mt-3', style={'color': 'crimson', 'fontWeight': 'bold'}),
    dcc.Interval(id='interval-anomaly', interval=60000, n_intervals=0)
])

# Validation layout to register all IDs for callback validation
app.validation_layout = html.Div([
    dcc.Location(id="url"),
    sidebar,
    content,
    home_layout,
    predict_layout,
    cost_layout,
    smart_layout,
    anomaly_layout
])

# ---------------- PAGE ROUTER ----------------
@app.callback(Output("page-content", "children"), Input("url", "pathname"))
def display_page(path):
    if path == '/predict': return predict_layout
    elif path == '/cost': return cost_layout
    elif path == '/smart': return smart_layout
    elif path == '/anomalies': return anomaly_layout
    return home_layout

# ---------------- CALLBACKS ----------------
@app.callback(Output('main-bar-graph', 'figure'),
              [Input('x-axis-choice', 'value'),
               Input('filter-day', 'value'),
               Input('filter-month', 'value'),
               Input('interval-dashboard', 'n_intervals')])
def update_main_bar_graph(xcol, day, month, _):
    df = load_data()
    if day: df = df[df['day_of_month'] == day]
    if month: df = df[df['month_of_year'] == month]
    if df.empty: return go.Figure()
    return px.bar(df, x=xcol, y='consumption', color='appliance')

@app.callback(Output('line-hourly-consumption', 'figure'),
              [Input('filter-day', 'value'),
               Input('filter-month', 'value'),
               Input('interval-dashboard', 'n_intervals')])
def update_line_chart(day, month, _):
    df = load_data()
    if day: df = df[df['day_of_month'] == day]
    if month: df = df[df['month_of_year'] == month]
    if df.empty: return go.Figure()
    avg_df = df.groupby('hour_of_day')['consumption'].mean().reset_index()
    return px.line(avg_df, x='hour_of_day', y='consumption')

@app.callback(Output('appliance-pie-chart', 'figure'),
              [Input('filter-day', 'value'),
               Input('filter-month', 'value'),
               Input('interval-dashboard', 'n_intervals')])
def update_pie_chart(day, month, _):
    df = load_data()
    if day: df = df[df['day_of_month'] == day]
    if month: df = df[df['month_of_year'] == month]
    if df.empty: return go.Figure()
    pie_data = df.groupby('appliance')['consumption'].sum().reset_index()
    return px.pie(pie_data, names='appliance', values='consumption', title="Consumption by Appliance")

@app.callback(Output('box-hourly-consumption', 'figure'),
              [Input('filter-day', 'value'),
               Input('filter-month', 'value'),
               Input('interval-dashboard', 'n_intervals')])
def update_box_plot(day, month, _):
    df = load_data()
    if day: df = df[df['day_of_month'] == day]
    if month: df = df[df['month_of_year'] == month]
    if df.empty: return go.Figure()
    return px.box(df, x='hour_of_day', y='consumption', points='all',
                  title="Hourly Consumption Distribution")

@app.callback(Output('heatmap-consumption', 'figure'),
              [Input('filter-month', 'value'),
               Input('interval-dashboard', 'n_intervals')])
def update_heatmap(month, _):
    df = load_data()
    if month:
        df = df[df['month_of_year'] == month]
    if df.empty: return go.Figure()
    heatmap_data = df.pivot_table(index='day_of_month', columns='hour_of_day', values='consumption', aggfunc='mean')
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data.values,
        x=heatmap_data.columns,
        y=heatmap_data.index,
        colorscale='YlOrRd',
        colorbar=dict(title='kWh')
    ))
    fig.update_layout(title='📊 Consumption Heatmap (Day vs Hour)', xaxis_title='Hour', yaxis_title='Day')
    return fig

@app.callback(Output('total-units', 'children'), Input('interval-dashboard', 'n_intervals'))
def update_total_units(_):
    df = load_data()
    total = df[df['day_of_month'] == df['day_of_month'].max()]['consumption'].sum()
    return f"{total:.2f} kWh"

@app.callback(Output('estimated-cost', 'children'), Input('interval-dashboard', 'n_intervals'))
def update_estimated_cost(_):
    df = load_data()
    total = df[df['day_of_month'] == df['day_of_month'].max()]['consumption'].sum()
    return f"₹{total * 8:.2f}"

@app.callback(Output('peak-hour', 'children'), Input('interval-dashboard', 'n_intervals'))
def update_peak_hour(_):
    df = load_data()
    return f"{df.groupby('hour_of_day')['consumption'].mean().idxmax()}:00"

@app.callback(Output('prediction-output', 'children'),
              [Input('3lv', 'value'), Input('5lv', 'value'), Input('7lv', 'value'),
               Input('temp', 'value'), Input('hour', 'value'), Input('dow', 'value'),
               Input('dom', 'value'), Input('moy', 'value')])
def predict_hourly(l3, l5, l7, t, h, dow, dom, moy):
    row = pd.DataFrame([[l3, l5, l7, t, h, dow, dom, moy]], columns=X.columns)
    pred = model.predict(scaler.transform(row))[0]
    return f"🔮 Predicted Consumption: {pred:.3f} units"

@app.callback(Output('next-day-output', 'children'),
              Input('predict-day-btn', 'n_clicks'),
              [State('3lv', 'value'), State('5lv', 'value'), State('7lv', 'value'),
               State('temp', 'value'), State('dow', 'value'),
               State('dom', 'value'), State('moy', 'value')])
def predict_day(n, l3, l5, l7, t, dow, dom, moy):
    if n == 0: return ""
    total = sum([model.predict(scaler.transform([[l3, l5, l7, t, hr, dow, dom, moy]]))[0] for hr in range(24)])
    return f"📅 Total Predicted: {total:.2f} units"

# Fixed: Remove interval-dashboard dependency from cost callback
@app.callback(Output('cost-estimate-output', 'children'),
              Input('unit-cost', 'value'))
def estimate_cost(unit):
    df = load_data()
    total = df[df['day_of_month'] == df['day_of_month'].max()]['consumption'].sum()
    return f"💸 Estimated Cost Today: ₹{total * unit:.2f}"

@app.callback(Output('smart-schedule-graph', 'figure'), Input('interval-smart', 'n_intervals'))
def smart_schedule(_):
    df = load_data()
    avg = df.groupby('hour_of_day')['consumption'].mean().reset_index()
    return px.area(avg, x='hour_of_day', y='consumption')

@app.callback(Output('smart-schedule-tips', 'children'), Input('interval-smart', 'n_intervals'))
def smart_tip(_):
    df = load_data()
    avg = df.groupby('hour_of_day')['consumption'].mean()
    best = avg.idxmin()
    return f"✅ Best time for heavy appliances: {best}:00"

@app.callback(Output('anomaly-graph', 'figure'), Input('interval-anomaly', 'n_intervals'))
def update_anomaly_chart(_):
    df = load_data()
    if df.empty: return go.Figure()
    daily = df.groupby('day_of_month')['consumption'].sum().reset_index()
    clf = IsolationForest(contamination=0.1)
    daily['anomaly'] = clf.fit_predict(daily[['consumption']])
    fig = px.line(daily, x='day_of_month', y='consumption', title='Daily Consumption with Anomalies')
    fig.add_trace(go.Scatter(x=daily[daily['anomaly'] == -1]['day_of_month'],
                             y=daily[daily['anomaly'] == -1]['consumption'],
                             mode='markers', marker=dict(color='red', size=10), name='Anomaly'))
    return fig

@app.callback(Output('recommendations-output', 'children'), Input('interval-anomaly', 'n_intervals'))
def tips(_):
    df = load_data()
    avg = df.groupby('hour_of_day')['consumption'].mean()
    top = avg.sort_values(ascending=False).head(2).index.tolist()
    return f"⚠️ Avoid heavy usage at: {top[0]}:00 & {top[1]}:00"

# ---------------- AUTO OPEN BROWSER ----------------
if __name__ == '__main__':
    def open_browser():
        time.sleep(1.5)
        webbrowser.open("http://127.0.0.1:8050/")
    threading.Thread(target=open_browser).start()
    app.run(debug=True)