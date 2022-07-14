import json
import os
import pickle
import random
import re
from typing import Dict, List, Tuple

from .utils import load_dataset, load_pickles
from .manager import DataManager
from .text import plot

from dash import Dash, dcc, html, dash_table
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
import fire
import numpy as np
import pandas as pd
import shap
from dash.dependencies import Input, Output, State
from plotly.graph_objects import Figure, Scatter
from whoosh.index import open_dir
from whoosh.qparser import MultifieldParser


def parser(path: str, query_string: str, field_list: List[str], page: int=1, limit: int=10) -> Tuple[int, List[Dict]]:
    """
    path: str
        path to directory contianing indexed data
    query: str
        search query
    field_list: List(str)
        list of fields to search, some combination of "report", "codeStr", and "code"
    limit: int or None
        number of results to return, set to None to return all

    returns tuple
    """
    ix = open_dir(path)
    mparser = MultifieldParser(field_list, schema=ix.schema)
    query = mparser.parse(query_string)

    with ix.searcher() as searcher:
        #search_out = searcher.search(query, limit=limit)
        search_out = searcher.search_page(query, page, pagelen=limit)
        out = []
        for sO in search_out:
            single_dict = {}
            for key in sO.keys():
                single_dict[key] = sO[key]
            out.append(single_dict)
        return len(search_out), out


def find_filtered_report(false_true: int, neg_pos: int, y: List[int], preds: List[int]) -> List[int]:
    """
    Returns filtered list of path reports

    false_true: int
        0->False, 1->True, 2->both
    neg_pos: int
        0->negative, 1->positive, 2->both
    y: List[int]
        array of correct codes for all reports for designated code
    preds: List[int]
        array of predictions for all reports for designated code

    List[int]
        list of indexes that meet filter specifications
    """
    if neg_pos == 2:
        # show both true and false
        value_indices = [i for i in range(len(y))]
    else:
        # find where y is equal to 0 (negative) or 1 (positive)
        value_indices = np.where(preds == neg_pos)[0]

    if false_true == 2:
        # show both positive and negative
        bool_indices = [i for i in range(len(y))]
    else:
        # generate list containing boolean for each prediction
        bool_values = y == preds

        # find in that list where predictions were either false or true
        bool_indices = np.where(bool_values == bool(false_true))[0]
    return np.intersect1d(value_indices, bool_indices)


def graph_info(data_obj, reportVal: int, hideCode: bool=False, model_val: int=-1, sort_by="prediction") -> Figure:
    # if multi output the ["proba"] is an array that is now inside array probas
    # if multi output the ["test"] is an index value
    # XXX
    if len(data_obj.results) == 1: #isinstance(probas[0], np.ndarray):
        probas = [probas[reportVal] for probas in data_obj.results[0]["pp"]["probas"]]
        preds = [preds[reportVal] for preds in data_obj.results[0]["pp"]["preds"]]
    else:
        probas = [result["pp"]["probas"][reportVal] for result in data_obj.results]
        preds = [result["pp"]["preds"][reportVal] for result in data_obj.results]

    classifications = [data_obj.allData["y"][code+" "].iloc[reportVal] for code in data_obj.codes]

    data = {"prediction": probas}
#     data["model"] = [i for i in range(len(probas))]
    data["code"] = data_obj.codes

    if not hideCode:
        data["status"] = ['True' if (prd == cls) else "False" for prd, cls in zip(preds, classifications)]
    else:
        data["status"] = ["*Hidden*" for i in range(len(probas))]

    # set color based off True/False/Hidden
    data["color"] = ['green' if data["status"][i] == "True" else 'red' if data["status"][i] == "False" else 'darkblue' for i in range(len(probas))]

    df = pd.DataFrame(data)
    # sort dataframe
    assert sort_by in ("code","prediction")
    df = df.sort_values(sort_by).reset_index(drop=True)

    fig = Figure()
    fig.update_yaxes(tickvals=[0, .5, 1], zeroline=False, range=[-.05, 1.05], linecolor="lightgray", gridcolor="gainsboro")
    if len(data_obj.codes) <= 10:
        fig.update_xaxes(showline=True, showgrid=False, zeroline=False, tickvals=df["code"], gridcolor="purple")#, ticktext=df["code"])
    else:
        fig.update_xaxes(showline=True, showgrid=False, zeroline=False, tickvals=[], gridcolor="purple")
    fig.update_layout(margin=dict(l=20, r=20, t=20, b=20),
                      legend_title="",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                      plot_bgcolor= "white",  # set plot background to white
                      xaxis_type = 'category',  # deactivate x-axis sorting
                      hoverlabel = dict(bgcolor="white", font_size=12),
                      #title="Individual Model Predictions",
                      yaxis_title="Prediction",
                      xaxis_title="Model")

    # construct hover text
    hoverText = [0 for _ in range(len(df))]
    for i in range(len(df)):
        hoverText[i] = (
#             f"Model No.={df['model'][i]}<br>"
            f"For CPT Code={df['code'][i]}<br>"
            f"Prediction={round(df['prediction'][i], 3)}<br>"
            f"Status={df['status'][i]}<br>"
        )
    # add points
    fig.add_trace(Scatter(x = df["code"],
                          y = df["prediction"],
                          mode = 'markers',
                          marker_color = df["color"],
                          marker_size  = 7,
                          hoverinfo = "text",
                          hovertext = hoverText))

    # draw lollipop lines
    for i in range(0, len(df)):
        fig.add_shape(type='line',
                      x0 = i, y0 = .5,
                      x1 = i,
                      y1 = df["prediction"][i],
                      line=dict(color='gray', width = 2),
                      layer='below')

    if hideCode:
        fig.update_layout(showlegend=False)

    return fig


def get_status(prediction: int, truth: int) -> str:
    """Return Status: True|False positive|negative"""
    part1 = str(prediction == truth)

    if prediction == 1:
        part2 = "positive"
    else:
        part2 = "negative"

    return f"\nStatus: {part1 + ' ' + part2}"


def initiate_app(port: int=8040, debug: bool=False):
    """
    port: int
        Local port.
    debug: bool
        Display debug on app.
    """
    dataset = load_dataset()
    codes = dataset["y"].columns.values
    codesClean = np.array([code[:-1] for code in dataset["y"].columns.values])

    with open("/dartfs/rc/nosnapshots/V/VaickusL-nb/EDIT_Students/projects/cpt_code_app_data/data/cpt_codes.json", "r") as f:
        codeDict = json.loads(f.read())
    
    # instantiate data manager
    d1 = DataManager()
    
    # create dict to store user preds
    user_assignments = {}

    # add models trained on whole reports to data manager
    dx_total = "total"
    d1.set(
        name="38 most common, total",
        dataset=dataset, 
        dx_total=dx_total, 
        path="/dartfs/rc/nosnapshots/V/VaickusL-nb/EDIT_Students/projects/cpt_code_app_data/data/total_code_models"
    )

    # add 5 code total model to data manager
    dx_total = "total"
    d1.set(
        name="primary codes, total", 
        dataset=dataset, 
        dx_total=dx_total, 
        path="/dartfs/rc/nosnapshots/V/VaickusL-nb/EDIT_Students/projects/cpt_code_app_data/data/total_pathologist_models"
    )

    # add models trained on only diagnostic section to data manager
    # on second thought, don't, dx needs to be retrained...
#     dx_total = "dx"
#     #d1.set(dx_total, dataset=dataset, dx_total=dx_total, path=os.path.join("code_models"), filename_stem="")
#     d1.set(dx_total, dataset=dataset, dx_total=dx_total, path=os.path.join("data/dx_code_models"), filename_stem="_rf_dx.pkl")

    report_index = 17
    # Build App
    app = Dash(__name__, external_stylesheets=[dbc.themes.SIMPLEX])

    app.layout = html.Div([
        dbc.Navbar(dbc.Container([
            dbc.Row(
                [
                    dbc.Col(dbc.NavbarBrand("EDIT NLP CPT demo")),
                    dbc.Col(html.Div(dcc.Dropdown(
                        id='algo-dropdown',
                        options=[{'value': name, 'label': f"{name.replace('_', ' ')} - {len(d1.data[name]['allData']['X'])} reports"} for i, name in enumerate(d1.data.keys())],
                        value=d1.current,
                        clearable=False
                    ), style={'width': '400px'})),
                ],
                align="center",
            ),
            dbc.Row([
                dbc.Col(html.Div(dbc.Label("Hide original assignment"), style={'padding': '5px', 'padding-top': '10px'})),
                dbc.Col(
                    dbc.Checklist(options=[{"value": 1}],
                            value=[1],
                            id="code-toggle",
                            switch=True),
                    width="auto",
                ),
            ],
            className="g-0 ms-auto flex-nowrap mt-3 mt-md-0",
            align="center",),
        ])),
        dbc.Col([
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.Label("Select CPT code:"),
                        dcc.Dropdown(
                            id='model-dropdown',
                            options= [{'label': f'SHAP values for code {code}', 'value': i} for i, code in enumerate(d1.codes)],
                            value='0',
                            clearable=False,
                            style={"color": "black", 'margin-bottom': 10}),
                        dbc.Label("Index of path report:"),
                        dbc.Input(id="path-num-input", type="number", value=report_index, debounce=True, style={'margin-bottom': 10}),
                        dbc.Label("All predictions"),
                        dcc.Graph(figure=graph_info(d1, report_index, sort_by="code"),
                                style={'display': 'inline-block', "width": "100%", "height": "350px", "border": 0},
                                id='scatter-graph'),
                    ], outline=True, color="primary", style={'height': "560px", "padding": ".5rem", "margin-bottom": "15px", "margin-top": "7.5px"}),
                ], width=4),
                dbc.Col([
                    dbc.Card([
                        dbc.Row([
                            dbc.Col(html.P(f"SHAP analysis for the prediction of CPT code {d1.codes[0]} on path report #{report_index}",
                                style={'textAlign': 'center'},
                                id='shap-plot-header')),
                            dbc.Col(dbc.Button("Open Larger", id="open-modal", n_clicks=0, color="success"), width=3)
                        ]),
                        html.Iframe(
                                    style={"width": "100%", "height": "390px", "border": 0, "backgroundColor": "white"},
                                    id="shap-plot"),
                        dcc.Loading(
                            id="loading-symbol",
                            type="default",
                            children=html.Div(id="loading-output")),
                        dcc.Textarea(id='info-block', readOnly=True, style={'minHeight': 105, 'width':"100%", "margin-top": "5px"}),
                    ], outline=True, color="primary", style={'height': "560px", "padding": ".5rem", "margin-bottom": "15px", "margin-top": "7.5px"}),
                ], width=8),
            ]),

            dbc.Collapse(dbc.Card(
                dbc.Form([
                    dbc.Row([
                        dbc.Col([
                            dbc.FormText("Enter all CPT codes that appear in the current report:"),
                        ], style={"margin-top": 2}),
                        dbc.Col([
                            dbc.Button("Download CPT Code Assignments CSV", id="download-button", size="sm", style={"float": "right"}),
                            dcc.Download(id="download-data"),
                        ])
                    ], style={'margin-bottom': 5}),
                    dbc.Row([
                        dcc.Dropdown(
                            id='code-dropdown',
                            options = [{"label": f"{c}: {d}", "value": c} for c, d in codeDict.items()],
                            placeholder = "Input codes here...",
                            multi=True,
                        ),
                    ]),
                    dbc.Label("Filter results:", id="form-text"),
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Restrict to:"),
                            dbc.RadioItems(
                                options=[
                                    {"label": "True predictions", "value": 1}, # these are set specifically for a later bitwise operation
                                    {"label": "False predictions", "value": 0},
                                    {"label": "Both", "value": 2},
                                ],
                                value=2,
                                id="accuracy-filter",
                                style={"margin-left": "10px"}
                            ),
                        ], width=2),
                        dbc.Col([
                            dbc.Label(" "),
                            dbc.RadioItems(
                                options=[
                                    {"label": "Positive only", "value": 1},
                                    {"label": "Negative only", "value": 0},
                                    {"label": "Both", "value": 2},
                                ],
                                value=2,
                                id="pos-neg-filter",
                                style={"margin-left": "10px"}
                            )
                        ], width=2),
                        dbc.Col([
                            dbc.Label("Using model trained to detect:"),
                            dcc.Dropdown(
                                id='filter-dropdown',
                                options = [{"label": f"{code}: {codeDict[code]}", "value": i} for i, code in enumerate(d1.codes)],
                                placeholder = "Input code here...",
                                style={"margin-left": "5px", "margin-right": "10px"}
                            ),
                        ], width=8)
                    ]),
                    dbc.Button("Next report", id="next-button", n_clicks=0, className="ml-auto", style={'float': 'right','margin': 'auto'}, color="success"),
                ]), outline=True, color="primary", style={"padding": ".5rem"}),
                id="collapse",
                is_open=True,
                style={"margin-bottom": "15px"}
            ),

            dbc.Card([
                dbc.Row([
                    dbc.Col(dbc.Label("Search by:")),
                    dbc.Col(
                        dbc.Checklist(
                            options=[
                                {"label": "Report", "value": "report"},
                                {"label": "Code", "value": "code"},
                                {"label": "Code description", "value": "codeStr"}
                            ],
                            value=["report"],
                            id="query-checklist",
                            inline=True,
                        ),
                    ),
                ]),
                dbc.Row([
                    dbc.Col([
                        dbc.Input(id="search-input", type="string", placeholder="Search for report"),
                        dash_table.DataTable(style_cell={'whiteSpace': 'normal', 'height': 'auto', 'color': 'black', 'textOverflow': 'ellipsis',},
                                            style_cell_conditional=[{'if': {'column_id': 'Codes'}, 'width': '15%'},
                                                                    {'if': {'column_id': 'Text'}, 'textAlign': 'left'}],
                                            page_current=0,
                                            page_size=5,
                                            page_action='custom',
                                            id="search-results")
                    ])
                ]),
            ], style={"padding": ".5rem"}, outline=True, color="primary"),


            html.Div(id="temp-area"),
            html.P(id='placeholder'),
            html.P(id='placeholder-2'),
            html.P(id='placeholder-3'),
            html.Div(id='plot-container', children=[]),
        ], width={"size": 10, "offset": 1}),
        dbc.Modal([
            dbc.ModalHeader(id='shap-plot-header-modal'),
            dbc.ModalBody(
                html.Iframe(
                    style={"width": "100%", "height": "65vh", "border": 0, "backgroundColor": "white"},
                    id="shap-plot-modal"),
            ),
            dbc.ModalFooter(
                dbc.Button("Close", id="close", className="ms-auto", n_clicks=0)
            ),
        ], id="modal", is_open=False, style={"max-width": "none", "width": "90%"},),
    ])


    @app.callback(
        Output('model-dropdown', 'value'),
        Output('model-dropdown', 'options'),
        Output('filter-dropdown', 'options'),
        Output('code-dropdown', 'options'),
        Input('scatter-graph', 'clickData'),
        Input('algo-dropdown', 'value'),
        State('model-dropdown', 'value'),
        )#prevent_initial_call=True)
    def updateModel(click_data, algo_value, value):
        """
        Callback to update current model in dropdown when changed through scatter plot
        :param Dict[List[Dict]]: clickData --> data from scatter plot containing which model was clicked
        :return int: value --> model value
        """
        print(algo_value)
        if algo_value != d1.current:
            d1.set(algo_value)
            value = 0
        elif click_data:
            value = d1.codes.index(click_data['points'][0]['x'])

        filter_options = [{"label": f"{code}: {codeDict[code]}", "value": i} for i, code in enumerate(d1.codes)]

        assignment_options = [{"label": f"{c}: {codeDict[c]}", "value": c} for c in d1.codes]

        options = [{'label': f'SHAP values for code {code}', 'value': i} for i, code in enumerate(d1.codes)]
        return int(value), options, filter_options, assignment_options

    # update info
    @app.callback(
        Output('info-block', 'value'),
        Input('model-dropdown', 'value'),
        Input('path-num-input', 'value'),
        Input('code-toggle', 'value'))
    def updateInfo(model_val, reportVal, hideCode):
        """Update information section"""

        model_val = output_index = int(model_val)

        # get correct codes
        correctCodesIndex =  np.where(d1.allData["y"].iloc[reportVal] == 1)[0]

        if len(d1.results) == 1:
            output_index = model_val
            model_val = 0
            correctCodes = [["88302 ","88304 ","88305 ","88307 ","88309 "][index][:-1] for index in correctCodesIndex]
            prediction = d1.results[model_val]["pp"]["preds"][output_index][reportVal]

        else:
            correctCodes = [codes[index][:-1] for index in correctCodesIndex]
            prediction = d1.results[model_val]["pp"]["preds"][reportVal]

        # calculate predictions
#         prediction = d1.results[model_val]["best_model"].predict(d1.allData['count_mat'][reportVal], output_margin=False)[0]        
        
        blockText = f"Prediction: {('does not contain', 'contains')[prediction]} code {d1.codes[output_index]}"
        if len(d1.results) != 1:
            blockText += get_status(prediction, d1.codes[model_val] in correctCodes)
        elif d1.results[0]["best_model"].objective == "multi:softprob":
            codes_5 = d1.codes
            correct_indices = np.where([code in correctCodes for code in codes_5])[0]
            if len(correct_indices) != 1:
                blockText += "\nPLEASE NOTE: This algorithm was only trained to predict on reports containing exactly one of the following codes:"
                blockText += f"88302, 88304, 88305, 88307, 88309. This report contains {len(correct_indices)} of those codes."
            if not hideCode:  #XXX: hmmm
                if len(correct_indices) == 1:
                    part = "correct billing relative to original coder"
                    if prediction < correct_indices[0]:
                        part = "underbilling relative to original coder"
                    if prediction > correct_indices[0]:
                        part = "overbilling relative to original coder"
                    blockText += f"\nStatus: {part}"
        if not hideCode:
            blockText += f"\nCorrect codes by code: {correctCodes}"
            blockText += f"\nCorrect codes by index: {correctCodesIndex}"


        if reportVal in d1.results[model_val]["splits"][d1.results[model_val]["best_fold"]][1]:
            blockText += f"\nReport #{reportVal} was in this models test set"
        else:
            blockText += f"\nReport #{reportVal} was in this models training set"

        return blockText

    # update widgets
    @app.callback(
        Output('shap-plot-header', 'children'),
        Output('shap-plot', 'srcDoc'),
        Output('loading-output', 'children'),
        Output('shap-plot-header-modal', 'children'),
        Output('shap-plot-modal', 'srcDoc'),
        Input('model-dropdown', 'value'),
        Input('path-num-input', 'value'))
    def updatePlot(model_val, reportVal):
        """
        Update section containing shap plot(s)
        """
        model_val = int(model_val)
        output_index = None
        if len(d1.results) == 1:
            output_index = model_val
            model_val = 0

        if model_val not in d1.explainerDict:
            # if the model has not yet been explained, explain it and add it the dictionary
            # TODO: explainerDict does not appear to be getting updated
            explainer = shap.TreeExplainer(d1.results[model_val]["best_model"])
            d1.explainerDict[model_val] = {"explainer": explainer}

        # create plot header
        if output_index:
            header = f"SHAP analysis for the prediction CPT code {d1.codes[output_index]} on path report #{reportVal}"
        else:
            header = f"SHAP analysis for the prediction CPT code {d1.codes[model_val]} on path report #{reportVal}"


        # note about implementation here, should just always pass pandas data series...
        # fetch the text of the pathology report, formatting if necessary
        pathText = d1.allData['X'].iloc[reportVal]
        if isinstance(pathText, pd.core.series.Series):
            text = ""
            for val in pathText.index:
                # check if section empty --> if empty do not include header
                if not pd.isna(pathText[val]):
                    text += f"!{val.replace(' ', '_')}! "  # can't use a space, have to split with something else
                    text += f"{pathText[val]} "  # add space at end
            report = text
        else:
            report = d1.allData['X'].iloc[reportVal]


        model = d1.results[model_val]["best_model"]
        sMR = d1.allData['count_mat'][reportVal]
        words = d1.allData['words']

        text_size = 18
        # we can pass output_index always because output_index is used only if the model is multi output
        src = plot(model, report, sMR, words, output_index=output_index, text_size=text_size)

        srcBig = src.replace(f"font-size: {text_size + 2}px", "font-size: 24px")  # replace headers
        srcBig = srcBig.replace(f"font-size: {text_size}px", "font-size: 22px")  # replace main
        srcBig = srcBig.replace(f"font-size: {text_size - 2}px", "font-size: 20px")  # replace supertext

        return header, src, None, header, srcBig

    @app.callback(
        Output('scatter-graph', 'figure'),
        Input('path-num-input', 'value'),
        Input('code-toggle', 'value'),
        Input('code-dropdown', 'options'),  # wait for output of updateModel
        State('algo-dropdown', 'value'),
        State('model-dropdown', 'value'))
    def updateGraph(reportVal, codeMode, _options, algo_val, model_val):
        fig = graph_info(d1, reportVal, bool(codeMode), model_val, sort_by="code")
        return fig

    @app.callback(
        Output('search-results', 'data'),
        Output('search-results', 'columns'),
        Output('search-results', 'page_count'),
        Input('search-input', 'value'),
        Input('search-results', "page_current"),
        Input('code-toggle', 'value'),
        State('query-checklist', 'value'))
    def search(query, page, hideCode, fields):
        if not query:
            return None, None, 0
        # dont forget about adjustable path
        numResults, search_out = parser("/dartfs/rc/nosnapshots/V/VaickusL-nb/EDIT_Students/projects/cpt_code_app_data/data/index", query, fields, page=page+1, limit=5)
        indices = [sO["index"] for sO in search_out]
        lenLimit = 500

        # TODO: look at all reports instead of subsect
        reports = [d1.allData["X"].iloc[index] for index in indices]
        texts = [txt[:lenLimit-3] + "..." if len(txt)>lenLimit else txt for txt in reports]
        df = pd.DataFrame([{"Text": text,
                            "Codes": ", ".join(codesClean[np.where(d1.allData["y"].iloc[index]==1)]),
                            "index": index} for index, text in zip(indices, texts)])
        return df.to_dict('records'), [{"name": head, "id": head}  for head in df.columns if head != "index" and not (head == "Codes" and hideCode)], -(numResults // -5)
    
    # save user predictions to dict
    @app.callback(
        Output('code-dropdown', 'placeholder'),  # basically a dummy ouptut
        Input('code-dropdown', 'value'),
        State('path-num-input', 'value'))
    def save_user_pred(predicted, currIndex):
        if not d1.current in user_assignments:
            user_assignments[d1.current] = {}
        user_assignments[d1.current][currIndex] = predicted
        return "Input codes here..."
    
    # update user assignments dropdown
    @app.callback(
        Output('code-dropdown', 'value'),
        Input('model-dropdown', 'value'),
        Input('path-num-input', 'value'))
    def update_user_assignments_dropdown(_model_val, report_val):
        user_assignments_to_return = []
        if d1.current in user_assignments and report_val in user_assignments[d1.current]:
            user_assignments_to_return = user_assignments[d1.current][report_val]
        return user_assignments_to_return
    
    @app.callback(
        Output("path-num-input", "value"),
        Output("next-button", "n_clicks"),
        Output('form-text', 'children'),
        Input('search-results', 'active_cell'),
        Input("next-button", "n_clicks"),
        State("accuracy-filter", "value"),
        State("pos-neg-filter", "value"),
        State("filter-dropdown", "value"),
        State('search-results', 'data'),
        State('path-num-input', 'value'))
    def getSearch(selected, next_button_clicks, filterAcc, filterPosNeg, filterModel, tableData, currIndex):
        """
        Trigger update of page if a search result clicked, if code hide switch toggled, or if next button
        """

        # if "next-button" clicked
        fText = "Filter results:"
        if next_button_clicks:

            # XXX: make this search all modules?
            if None == filterModel:
                return currIndex + 1, 0, fText + " showing prediction"
            fText += f" showing {('false', 'true', '')[filterAcc]} {('negative', 'positive', 'prediction')[filterPosNeg]}"

            filter_kwargs = dict(
                false_true = filterAcc,
                neg_pos = filterPosNeg,
                y = d1.allData["y"][d1.codes[filterModel] + " "],
            )

            if d1.results[0]["best_model"].objective == "multi:softprob":
                filter_kwargs["preds"] = d1.results[0]["pp"]["preds"][filterModel]
            else:
                filter_kwargs["preds"] = d1.results[filterModel]["pp"]["preds"]
            indices = find_filtered_report(**filter_kwargs)

            index_of_index, found_index = 0, 0
            for i, newIndex in enumerate(indices):
                if (newIndex > currIndex):
                    index_of_index, found_index = i, newIndex
                    break

            return found_index, 0, f"{fText} {index_of_index+1}/{len(indices)}"

        elif selected:  # report selected from search
            index = tableData[selected["row"]]["index"]
            return index, 0, fText
        else:  # handle first call (nothing selected)
            raise PreventUpdate
    
    @app.callback(
        Output('search-input', 'disabled'),
        Output('search-input', 'placeholder'),
        Input('algo-dropdown', 'value'))
    def update_searchbox(algo_value):
        if algo_value == "38 most common, total":
            return False, "Search for report"
        else:
            return True, "Corpus search not available for this model"

    @app.callback(
        Output("collapse", "is_open"),
        Input("code-toggle", "value"))
    def codeSearchCollapse(hideCode):
        """toggle collapsable code search and select"""
        return bool(hideCode)


    @app.callback(
        Output("modal", "is_open"),
        [Input("open-modal", "n_clicks"), Input("close", "n_clicks")],
        [State("modal", "is_open")])
    def toggle_modal(n1, n2, is_open):
        if n1 or n2:
            return not is_open
        return is_open
    
    @app.callback(
        Output("download-data", "data"),
        Input("download-button", "n_clicks"))
    def download_assignments(n_clicks):
        if not n_clicks:
            raise PreventUpdate
        return dcc.send_data_frame(pd.DataFrame(user_assignments).to_csv, "user_assignments.csv")


    # Run app and display result in the notebook
    app.run_server(host="localhost", port=port, debug=debug)

def main():
    fire.Fire(initiate_app)
    
if __name__ == "__main__":
    main()
