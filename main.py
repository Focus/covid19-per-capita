from countryinfo import CountryInfo
import os
import numpy as np
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime, timezone

LAST_DATA_PULL = datetime.now(timezone.utc)
REFRESH_EVERY_SECONDS = 60 * 60 * 4  # 4 hours
DEFAULT_COUNTRIES = ['United Kingdom', 'Turkey']
BASE_DIR = os.path.dirname(__file__)
EXCLUSIONS = [
    ('Turkey', '2020-12-10', '2020-12-10')
]

app = dash.Dash(
    __name__,
    meta_tags=[
        {
            'name': 'viewport',
            'content': 'width=device-width, initial-scale=1.0'
        }
    ],
)
server = app.server


def apply_exclusions(df):
    exclude = pd.Series(False, index=df.index)
    for country, start, end in EXCLUSIONS:
        exclude |= (df['country'] == country) & \
            (df['date'] >= start) & \
            (df['date'] <= end)
    df.loc[exclude, 'new_cases'] = np.nan
    return df


def load_data():
    csv_file = (
        'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master'
        '/csse_covid_19_data/csse_covid_19_time_series/time_series_'
        'covid19_confirmed_global.csv'
    )
    df = pd.read_csv(csv_file)
    df.drop(['Province/State', 'Lat', 'Long'], axis=1, inplace=True)
    df = df.melt(id_vars=['Country/Region'],
                 var_name='date',
                 value_name='infected')
    df['date'] = pd.to_datetime(df['date'])

    df.columns = ['country', 'date', 'infected']
    df = df.groupby(['country', 'date']).sum().reset_index()
    df.sort_values(['date', 'country'], inplace=True)

    df['new_cases'] = df.groupby('country')['infected'].transform(
        lambda x: x - x.shift(1)
    )

    df = apply_exclusions(df)

    df['new_cases_per_mil'] = np.nan
    no_info_countries = []
    for country, frame in df.groupby('country'):
        try:
            country_info = CountryInfo(country)
            population = country_info.population()
            mask = df['country'] == country
            df.loc[mask, 'new_cases_per_mil'] = 1e6 * df.loc[mask, 'new_cases']
            df.loc[mask, 'new_cases_per_mil'] /= population
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
        frame['rolled'] = frame.groupby('country')['new_cases_per_mil']\
            .transform(
                lambda x: x.dropna().rolling(rolling_window).mean().reindex(x.index)
            )
    else:
        frame['rolled'] = frame['new_cases_per_mil']

    fig = go.Figure()
    for country in countries:
        country_frame = frame[frame['country'] == country]
        fig.add_trace(go.Line(
            x=country_frame['date'],
            y=country_frame['rolled'],
            name=country
        ))

    fig.update_layout(
        title=f'{rolling_window}-day moving average of new COVID19 cases per million',
        xaxis_title='',
        yaxis_title='',
        legend=dict(
            orientation="h",
        ),
        margin=dict(
            r=0,
            l=0,
        )
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
                marks={
                    0: {'label': 'No averaging'},
                    7: {'label': '7'},
                    14: {'label': '14'},
                    28: {'label': '28'},
                }),
        ]
    ),

    html.Div(children="""
    Data source: https://github.com/CSSEGISandData/COVID-19
    """)

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
    global df
    global LAST_DATA_PULL
    now = datetime.now(timezone.utc)
    if (now - LAST_DATA_PULL).seconds > REFRESH_EVERY_SECONDS:
        df = load_data()
        LAST_DATA_PULL = now
    return (plot_frame(df,
                       countries=countries,
                       rolling_window=window),
            f'Rolling window: {window}')


if __name__ == '__main__':
    app.run_server(debug=True)
