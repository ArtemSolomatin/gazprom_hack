from flask import Flask
from net import *
import numpy as np
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import random
import string

app = Flask(__name__)

# PREFIX = './static/'
columns = ['GZ3', 'bk', 'NKTR', 'NKTD', 'GZ1', 'DGK', 'ALPS']


@app.route('/', methods=['GET', 'POST'])
def get_graph(data_path, model):

    filename = randomString() + '.html'
    data = pd.read_csv(data_path)
    prediction = predict_by_df(data, model)
    fig_html = get_interactive_graph(data, prediction)
    save_fig(fig_html, filename)
    return '../static/' + filename


def save_fig(fig_html, filename):
    with open('static/' + filename, 'w+') as file:
        file.write(fig_html)


def randomString(stringLength=10):
    """Generate a random string of fixed length """
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(stringLength))


def get_income_oil(targets, depth):
    price = 4.15
    rho = 860
    S = 100
    porosity = 0.7
    diff = np.concatenate([np.asarray([0.0895]), np.diff(depth)])
    return porosity * price * rho * S * np.sum(targets * diff)


def get_expected_income_oil(probs, depth):
    price = 4.15
    rho = 860
    S = 100
    porosity = 0.7
    diff = np.concatenate([np.asarray([0.0895]), np.diff(depth)])
    return porosity * price * rho * S * np.sum(probs * diff)


def get_costs_research(features, depth):
    costs = {'bk': 2450,
             'GZ1': 2050,
             'GZ2': 2050,
             'GZ3': 2050,
             'GZ4': 2050,
             'GZ5': 2050,
             'GZ7': 2050,
             'DGK': 1300,
             'NKTD': 2050,
             'NKTM': 2050,
             'NKTR': 2050,
             'ALPS': 1150}
    cost = 0
    diff = np.diff(depth)
    for fea in features:
        cost += costs[fea]
    return np.sum(cost * diff)


def get_check_costs(depth, cost):
    diff = np.diff(depth)
    return np.sum(cost * diff)


def get_overall_income(goals, depth, features, cost):
    research_costs = get_costs_research(features, depth)
    oil_income = get_income_oil(goals, depth)
    check_costs = get_check_costs(depth, cost)
    income = oil_income - check_costs - research_costs
    return income


def get_overall_expected_income(probs, depth, features, cost):
    research_costs = get_costs_research(features, depth)
    oil_income = get_expected_income_oil(probs, depth)
    check_costs = get_check_costs(depth, cost)
    income = oil_income - check_costs - research_costs
    return income


def get_interactive_graph(df, probs):
    fig = make_subplots(rows=1, cols=len(columns) + 2, shared_yaxes=True, subplot_titles=tuple(columns + ['Probs', 'Targets']))
    df = df.sort_values('depth, m')
    depth = df['depth, m'].values
    for i, name in enumerate(columns):
        feature = df[name].values
        fig.add_trace(
            go.Scatter(x=feature, y=depth, name=name),
            row=1, col=i + 1)

    fig.add_trace(go.Scatter(
        x=probs,
        y=depth, name='Probs'), row=1, col=i + 2)
    slide_modes = 12

    # skip_first = True
    for threshold in np.linspace(probs.min(), probs.max(), slide_modes):
        # if skip_first:
        #     skip_first = False
        #     continue

        fig.add_trace(go.Heatmap(
            z=np.asarray(probs > threshold)*1,
            x=np.ones(depth.size),
            y=depth, name='Thershold:' + str(threshold),
            visible=False), row=1, col=i + 3)

    # fig.data[13+5].visible=True
    thresholds = []
    for j in range(slide_modes - 1):
        thresh = dict(
            method="restyle",
            args=[{"visible": [True] * (len(columns) + 2) + [False] * slide_modes,
                   'text.title': 'asd'}]
        )
        thresh['args'][0]['visible'][8 + j] = True
        thresholds.append(thresh)

    sliders = [dict(
        active=12,
        currentvalue={"prefix": "Threshold: "},
        pad={"t": 50},
        steps=thresholds
    )]
    expected_income = get_overall_income(probs, depth, columns, 1000)
    fig.update_layout(title=go.layout.Title(
        text="Expected income: %d rubles" % (expected_income),
        xref="paper",
        x=0))

    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(showticklabels=False)
    fig.update_layout(showlegend=False, height=710, width=1200, plot_bgcolor='rgba(0,0,0,0)', sliders=sliders)
    # fig.show()
    return fig.to_html()