from countryinfo import CountryInfo
import os
import numpy as np
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
import pandas as pd

DEFAULT_COUNTRIES = ['United Kingdom', 'Turkey']
BASE_DIR = os.path.dirname(__file__)

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
with open(os.path.join(BASE_DIR, 'index.html'), 'r') as fh:
    app.index_string = fh.read()

def load_data():
    csv_file = ('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master'
            '/csse_covid_19_data/csse_covid_19_time_series/time_series_'
            'covid19_confirmed_global.csv')

    df = pd.read_csv(csv_file)
    df.drop(['Province/State', 'Lat', 'Long'], axis=1, inplace=True)
    df = df.melt(id_vars=['Country/Region'],
                 var_name='date',
                 value_name='infected')
    df['date'] = pd.to_datetime(df['date'])

    df.columns = ['country', 'date', 'infected']
    df = df.groupby(['country', 'date']).sum().reset_index()
    df.sort_values(['date', 'country'], inplace=True)

    df['new_cases'] = df.groupby('country')['infected'].transform(lambda x: x - x.shift(1))
    df['new_cases_per_mil'] = np.nan
    no_info_countries = []
    for country, frame in df.groupby('country'):
        try:
            country_info = CountryInfo(country)
            population = country_info.population()
            mask = df['country'] == country
            df.loc[mask, 'new_cases_per_mil'] = 1e6 * df['new_cases'] / population
        except KeyError:
            no_info_countries.append(country)
            pass
    df = df[~df['country'].isin(no_info_countries)]
    return df


def plot_frame(df, rolling_window=14, countries=None):
    if countries is None:
        countries = []
    if len(countries) == 0:
        return {}
    frame = df[df['country'].isin(countries)].copy()
    if rolling_window > 0:
        frame['rolled'] = frame.groupby('country')['new_cases_per_mil'].transform(
            lambda x: x.rolling(rolling_window).mean()
        )
    else:
        frame['rolled'] = frame['new_cases_per_mil']
    fig = px.line(frame,
                  x='date',
                  y='rolled',
                  color='country',
                  category_orders={'country': countries})
    fig.update_layout(
        title=f'{rolling_window}-day moving average of new COVID19 cases per million',
        xaxis_title='',
        yaxis_title=f'{rolling_window}-day moving average of new cases per million'
    )
    return fig


df = load_data()
fig = plot_frame(df, countries=DEFAULT_COUNTRIES)

app.layout = html.Div(children=[
    html.H1(children='COVID19 cases across countries'),

    dcc.Graph(
        id='covid-graph',
        figure=fig
    ),

    dcc.Dropdown(
        id='country-select',
        options=[{'label': country, 'value': country}
                 for country in df['country'].unique()],
        value=DEFAULT_COUNTRIES,
        multi=True
    ),

    html.Div(
        children=[
            html.Div(children='Rolling window: 14',
                     id='rolling-label'),
        dcc.Slider(
            id='rolling-window',
            min=0,
            max=30,
            step=1,
            value=14,
        ),
        ]
    ),

    html.Div(children='''
    Data source: 
    https://github.com/CSSEGISandData/COVID-19
    '''),

])

@app.callback(
    [
        dash.dependencies.Output('covid-graph', 'figure'),
        dash.dependencies.Output('rolling-label', 'children')
    ],
    [
        dash.dependencies.Input('country-select', 'value'),
        dash.dependencies.Input('rolling-window', 'value')
    ]
)
def update_output(countries, window):
    return (plot_frame(df,
                       countries=countries[::-1],
                       rolling_window=window),
            f'Rolling window: {window}')


if __name__ == '__main__':
    app.run_server(debug=True)
