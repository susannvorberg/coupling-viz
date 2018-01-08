# -*- coding: utf-8 -*-
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import flask
import os
import numpy as np
import base64
import pandas as pd
import json
import contact_prediction.utils.io_utils as io
import contact_prediction.utils.utils as u
import contact_prediction.utils.pdb_utils as pdb
import contact_prediction.utils.alignment_utils as au
import contact_prediction.utils.benchmark_utils as bu
import contact_prediction.utils.ccmraw as raw
import contact_prediction.utils.plot_utils as plot
import contact_prediction.plotting.plot_alignment_aminoacid_distribution as alignment_plot
import contact_prediction.plotting.plot_pairwise_aa_freq as pairwise_aa_plot
import contact_prediction.plotting.plot_alignment_coverage as aligncov
import contact_prediction.plotting.plot_precision_vs_rank as prec_plot

server = flask.Flask(__name__)
server.secret_key = os.environ.get('secret_key', 'secret')
app = dash.Dash(name = __name__, server = server)
app.config.supress_callback_exceptions = True

app.css.config.serve_locally = True
app.scripts.config.serve_locally = True


############################################################
# Define Layout
############################################################

app.layout = html.Div([
    ##Alignment Stats Top Right
    html.Div(id="alignment_stats",
             style={'position': 'absolute', 'left': '25%', 'top': '3%', 'width': '63%', 'height': '10%'}),

    ## tabs and Output Right
    html.Div([
        dcc.Tabs(
            tabs=[{'label': 'Alignment', 'value': 1},
                  {'label': 'Pairwise AA Freq', 'value': 2},
                  {'label': 'Contact Maps', 'value': 3},
                  {'label': 'Precision', 'value': 4},
                  {'label': 'Coupling Matrices', 'value': 5}],
            value=1,
            id='tabs'
        ),
        html.Div(id='tab-output-1',style={'display': 'block'}),
        html.Div(id='tab-output-2',style={'display': 'none'}),
        html.Div(id='tab-output-3',style={'display': 'none'}),
        html.Div(id='tab-output-4',style={'display': 'none'}),
        html.Div(id='tab-output-5',style={'display': 'none'}),
    ], style={'position': 'absolute', 'left': '25%', 'top': '15%', 'width': '63%', 'height': '80%'}),


    ## Menu - left

    html.Div([

        dcc.Upload(
            id='upload-alignment',
            children=[
                html.Button('Upload Alignment File',
                            style={'background-color': 'white', 'border': '2px solid #4CAF50', 'color' : 'black', 'font-size': '16px', 'padding': '15px 32px', 'border-radius': '4px'})
            ],
            multiple=False
        ),
        html.Br(),
        dcc.Upload(
            id='upload-pdb',
            children=[
                html.Button('Upload PDB File',
                            style={'background-color': 'white', 'border': '2px solid #4CAF50', 'color': 'black',
                                   'font-size': '16px', 'padding': '15px 32px', 'border-radius': '4px'})
            ],
            multiple=False
        ),
        html.Br(),
        dcc.Upload(
            id='upload-braw',
            children=[
                html.Button('Upload binary raw coupling File',
                            style={'background-color': 'white', 'border': '2px solid #4CAF50', 'color': 'black',
                                   'font-size': '16px', 'padding': '15px 32px', 'border-radius': '4px'})
            ],
            multiple=False
        ),
        html.Br(),
        html.Br(),



        html.Div(
            id='residue_pair_ids',
            children = [
                html.Div(
                    children = [
                        html.Label("Residue Pair i:  ", style={'font-size': '16px'}),
                        dcc.Dropdown(
                            options=[{'label': str(i), 'value': str(i), 'disabled': 'True'} for i in range(1,2)],
                            id="res_i",
                            value="1")
                    ],
                    style={'display': 'inline', 'width': '10'}
                ),
                html.Div(
                    children = [
                        html.Label("Residue Pair j:  ", style={'font-size': '16px'}),
                        dcc.Dropdown(
                            options=[{'label': str(i), 'value': str(i), 'disabled': 'True'} for i in range(2,3)],
                            id="res_j",
                            value="2")
                    ],
                    style={'display': 'inline', 'max-width': '10'}
                ),
                html.Br()
            ],
            style={'display': 'none', 'width': '10'}
        ),


        html.Div(
            id="coupling_matrix_options",
            children = [
                html.Label("Apply local correction to coupling values:  ", style={'font-size': '16px'}),
                dcc.RadioItems(
                    id='coupling_matrix_correction',
                    options=[
                        {'label': 'no correction', 'value': 'no'},
                        {'label': 'Entropy Correction', 'value': 'ec'},
                        {'label': 'Count Correction', 'value': 'cc'}
                    ],
                    value='no',
                    labelStyle={'display': 'block'}
                ),
                html.Br()
            ],
            style={'display': 'none'}
        ),

    html.Div(
        id="contact_map_correction",
        children = [
            html.Label("Apply correction to contact score:  ", style={'font-size': '16px'}),
            dcc.RadioItems(
                id='contact_score_correction',
                options=[
                    {'label': 'no correction', 'value': 'no'},
                    {'label': 'APC', 'value': 'apc'},
                    {'label': 'Entropy Correction', 'value': 'ec'},
                    {'label': 'Count Correction', 'value': 'cc'}
                ],
                value='apc',
                labelStyle={'display': 'block'}
            ),
            html.Br()
        ],
        style={'display': 'none'}
        ),

        html.Div(
            id='contact_map_options',
            children = [
                html.Label("Sequence separation:  ", style={'font-size': '16px'}),
                dcc.Slider(
                    id='sequence_separation',
                    min=2,
                    max=12,
                    step=2,
                    value=6,
                    marks={i: str(i) for i in range(2,12+1,2)}
                ),
                html.Br(),
                html.Label("Contact Threshold:  ", style={'font-size': '16px'}),
                dcc.Slider(
                    id='contact_threshold',
                    min=4,
                    max=14,
                    step=2,
                    value=8,
                    marks={i: str(i) for i in range(4,14+1,2)}
                )
            ],
            style={'display': 'none'}
        )

    ], style={'position': 'absolute', 'left': '3%', 'top': '3%', 'width' : '20%'}),



    # Hidden div inside the app that stores the intermediate value
    html.Div(id='protein_alignment', style={'display': 'none'}),
    html.Div(id='protein_braw', style={'display': 'none'}),
    html.Div(id='protein_pdb', style={'display': 'none'})

])







############################################################
# Reactivity - load data
############################################################

@app.callback(Output('protein_alignment', 'children'),
                [Input('upload-alignment', 'contents'),
                 Input('upload-alignment', 'filename')]
              )
def load_alignment_data(alignment_contents_list, alignment_name):

    protein_alignment_dict = {}

    if alignment_contents_list is not None:

        protein_alignment_dict['protein_name'] = alignment_name.split(".")[0]
        protein_alignment_dict['alignment_filename'] = alignment_name

        content_type, content_string = alignment_contents_list.split(',')
        decoded_string = base64.decodestring(content_string)
        decoded_split_str = decoded_string.split("\n")

        alignment = np.array([[io.AMINO_INDICES[c] for c in x.strip()] for x in decoded_split_str[:-1]], dtype=np.uint8)

        protein_alignment_dict['N'] = alignment.shape[0]
        protein_alignment_dict['L'] = alignment.shape[1]
        protein_alignment_dict['alignment'] = alignment.reshape(protein_alignment_dict['N'] * protein_alignment_dict['L']).tolist()

    return json.dumps(protein_alignment_dict)


@app.callback(Output('protein_pdb', 'children'),
              [Input('upload-pdb', 'contents'),
               Input('upload-pdb', 'filename')],
              [State('protein_alignment', 'children')]
              )
def load_pdb_data(pdb_contents_list, pdb_name, protein_alignment_json):

    protein_pdb_dict = {}

    if pdb_contents_list is not None:

        protein_pdb_dict['protein_name'] = pdb_name.split(".")[0]
        protein_pdb_dict['pdb_filename'] = pdb_name

        content_type, content_string = pdb_contents_list.split(',')
        decoded_string = base64.decodestring(content_string)

        f = open('./tmp.pdb', 'w')
        f.write(decoded_string)
        f.close()

        L=None
        protein_alignment_dict = json.loads(protein_alignment_json)
        if 'L' in protein_alignment_dict:
            L = protein_alignment_dict['L']

        distance_map = pdb.distance_map('./tmp.pdb', L=L, distance_definition="Cb")
        protein_pdb_dict['distance_map'] = distance_map.reshape(L * L).tolist()

    return json.dumps(protein_pdb_dict)


@app.callback(Output('protein_braw', 'children'), [Input('upload-braw', 'contents'), Input('upload-braw', 'filename')])
def load_braw_data(braw_contents_list, braw_name):

    protein_braw_dict = {}

    if braw_contents_list is not None:
        protein_braw_dict['protein_name'] = braw_name.split(".")[0]
        protein_braw_dict['braw_filename'] = braw_name

        content_type, content_string = braw_contents_list.split(',')
        decoded_string = base64.decodestring(content_string)

        f = open('./tmp.braw.gz', 'w')
        f.write(decoded_string)
        f.close()

        braw = raw.parse_msgpack('./tmp.braw.gz')
        L = braw.ncol

        protein_braw_dict['meta'] = braw.meta

        protein_braw_dict['x_single'] = braw.x_single[:, :20].reshape(L * 20).tolist()
        protein_braw_dict['x_pair'] = braw.x_pair[:, :, :20, :20].reshape(L * L * 20 * 20).tolist()

    return json.dumps(protein_braw_dict)



############################################################
# Reactivity - Stats
############################################################

@app.callback(Output('alignment_stats', 'children'),
              [Input('protein_alignment', 'children'),
               Input('protein_pdb', 'children'),
               Input('protein_braw', 'children')])
def update_alignment_stats(protein_alignment_json, protein_pdb_json, protein_braw_json):

    protein_alignment_dict = json.loads(protein_alignment_json)
    protein_braw_dict = json.loads(protein_braw_json)
    protein_pdb_dict = json.loads(protein_pdb_json)

    protein_name = ""
    if 'protein_name' in protein_alignment_dict:
        protein_name = protein_alignment_dict['protein_name']
    elif 'protein_name' in protein_pdb_dict:
        protein_name = protein_pdb_dict['protein_name']
    elif 'protein_name' in protein_braw_dict:
        protein_name = protein_braw_dict['protein_name']

    L = ""
    if 'L' in protein_alignment_dict:
        L = protein_alignment_dict['L']
    elif 'meta' in protein_braw_dict:
        L = u.find_dict_key('ncol', protein_braw_dict['meta']['workflow'][0])

    N = ""
    if 'N' in protein_alignment_dict:
        N = protein_alignment_dict['N']
    elif 'meta' in protein_braw_dict:
        N = u.find_dict_key('nrow', protein_braw_dict['meta']['workflow'][0])

    Neff = ""
    if 'meta' in protein_braw_dict:
        Neff = np.round(u.find_dict_key('neff', protein_braw_dict['meta']['workflow'][0]), decimals=3)

    Diversity=""
    if L != "" and N != "":
        Diversity = np.round(np.sqrt(int(N)) / int(L), decimals=3)

    if 'alignment' in protein_alignment_dict:
        status_1 ="Alignment file {0} successfully loaded.".format(protein_alignment_dict['alignment_filename'])
    else:
        status_1 = "No alignment file loaded!"

    if 'distance_map' in protein_pdb_dict:
        status_2 ="PDB file {0} successfully loaded.".format(protein_pdb_dict['pdb_filename'])
    else:
        status_2 = "No PDB file loaded!"

    if 'x_pair' in protein_braw_dict:
        status_3 ="Binary raw file {0} successfully loaded.".format(protein_braw_dict['braw_filename'])
    else:
        status_3 = "No binary raw file loaded!"

    status_div = html.Div(
        children=[html.P(status_1), html.P(status_2), html.P(status_3)],
        style={'position': 'absolute', 'left': '0%', 'top': '0%', 'width' : '50%', 'text-align': 'left'}
    )

    header_1 = html.H3("Protein {0}".format(protein_name))
    table = html.Table([
        html.Tr([
            html.Td("protein length", style={'padding': '5'}),
            html.Td("number of sequences", style={'padding': '5'}),
            html.Td("Neff", style={'padding': '5'}),
            html.Td("Diversity", style={ 'padding': '5'})
        ], style={'background': 'white', 'font-weight': 'bold'}),
        html.Tr([
            html.Td(L, style={'padding': '5'}),
            html.Td(N, style={'padding': '5'}),
            html.Td(Neff, style={ 'padding': '5'}),
            html.Td(Diversity, style={'padding': '5'})
        ], style={'background': 'white', 'font-weight': 'normal'})
    ], style={'border-collapse': 'collapse', 'margin-left': 'auto', 'margin-right': 'auto'})

    statistics_div = html.Div(
        children=[header_1, table],
        style={'position': 'absolute', 'left': '50%', 'top': '0%', 'width': '50%', 'text-align': 'center'}
    )

    return html.Div([status_div, statistics_div])

############################################################
# Menu display according to Tab
############################################################

@app.callback(Output('res_i', 'options'), [Input('protein_alignment', 'children')])
def update_res_i(protein_alignment_json):
    protein_alignment_dict=json.loads(protein_alignment_json)

    if 'L' in protein_alignment_dict:
        dropdown_options = [{'label': str(i), 'value': str(i)} for i in range(1, protein_alignment_dict['L']-1)]
    else:
        dropdown_options = [{'label': '1', 'value': '1'}]
    return(dropdown_options)


@app.callback(Output('res_j', 'options'), [Input('res_i', 'value'), Input('res_i', 'options')])
def update_res_j(value, res_i_options):
    L = len(res_i_options)
    dropdown_options = [{'label': str(i), 'value': str(i)} for i in range(int(value)+1, int(L)+2)]
    return(dropdown_options)


@app.callback(Output('contact_map_correction', 'style'), [Input('tabs', 'value')])
def adjust_menu(value):

    if value == 1:
        return {'display': 'none'}
    elif value == 2:
        return {'display': 'none'}
    elif value == 3:
        return {'display': 'block'}
    elif value == 4:
        return {'display': 'none'}
    elif value == 5:
        return {'display': 'none'}

@app.callback(Output('contact_map_options', 'style'), [Input('tabs', 'value')])
def adjust_menu(value):

    if value == 1:
        return {'display': 'none'}
    elif value == 2:
        return {'display': 'none'}
    elif value == 3:
        return {'display': 'block'}
    elif value == 4:
        return {'display': 'block'}
    elif value == 5:
        return {'display': 'none'}

@app.callback(Output('residue_pair_ids', 'style'), [Input('tabs', 'value')])
def adjust_menu(value):

    if value == 1:
        return {'display': 'none'}
    elif value == 2:
        return {'display': 'block'}
    elif value == 3:
        return {'display': 'none'}
    elif value == 4:
        return {'display': 'none'}
    elif value == 5:
        return {'display': 'block'}

@app.callback(Output('coupling_matrix_options', 'style'), [Input('tabs', 'value')])
def adjust_menu(value):

    if value == 1:
        return {'display': 'none'}
    elif value == 2:
        return {'display': 'none'}
    elif value == 3:
        return {'display': 'none'}
    elif value == 4:
        return {'display': 'none'}
    elif value == 5:
        return {'display': 'block'}


############################################################
# Switch Tab Visibility
############################################################


@app.callback(Output('tab-output-1', 'style'), [Input('tabs', 'value')])
def switch_visibility_tab_1(value):
    if value == 1:
        return {'display': 'block', 'align':'center'}
    else:
        return {'display': 'none'}

@app.callback(Output('tab-output-2', 'style'), [Input('tabs', 'value')])
def switch_visibility_tab_2(value):
    if value == 2:
        return {'display': 'block', 'align':'center'}
    else:
        return {'display': 'none'}

@app.callback(Output('tab-output-3', 'style'), [Input('tabs', 'value')])
def switch_visibility_tab_3(value):
    if value == 3:
        return {'display': 'block', 'align':'center'}
    else:
        return {'display': 'none'}

@app.callback(Output('tab-output-4', 'style'), [Input('tabs', 'value')])
def switch_visibility_tab_4(value):
    if value == 4:
        return {'display': 'block', 'align': 'center'}
    else:
        return {'display': 'none'}

@app.callback(Output('tab-output-5', 'style'), [Input('tabs', 'value')])
def switch_visibility_tab_5(value):
        if value == 5:
            return {'display': 'block', 'align': 'center'}
        else:
            return {'display': 'none'}
############################################################
# Tab Display
############################################################


@app.callback(Output('tab-output-1', 'children'),
              [Input('tabs', 'value'),
               Input('protein_alignment', 'children')])
def display_tab_1(value, protein_alignment_json):

    if value == 1 and protein_alignment_json is not None:

        protein_alignment_dict = json.loads(protein_alignment_json)

        figure={}
        h2=""

        if 'alignment' in protein_alignment_dict:
            alignment = np.array(protein_alignment_dict['alignment'], dtype=np.uint8)
            alignment = alignment.reshape((protein_alignment_dict['N'], protein_alignment_dict['L']))

            figure = alignment_plot.plot_amino_acid_distribution_per_position(alignment, "", plot_file=None, freq=False)
            h2=html.H2("Distribution of Amino Acids per position in alignment")

        graph_element = dcc.Graph( id='graph', figure=figure, style={'height': 800} )
        return html.Div([h2, graph_element], style={'text-align': 'center'})


@app.callback(Output('tab-output-2', 'children'),
              [Input('tabs', 'value'),
               Input('protein_alignment', 'children'),
               Input('res_i', 'value'),
               Input('res_j', 'value')
               ])
def display_tab_2(value, protein_alignment_json, residue_i, residue_j):

    if value == 2 and protein_alignment_json is not None:

        protein_alignment_dict = json.loads(protein_alignment_json)
        figure = {}

        if 'alignment' in protein_alignment_dict:
            alignment = np.array(protein_alignment_dict['alignment'], dtype=np.uint8)
            alignment = alignment.reshape((protein_alignment_dict['N'], protein_alignment_dict['L']))
            protein_name = protein_alignment_dict['protein_name']

            figure = pairwise_aa_plot.plot_aa_frequencies(
                alignment, protein_name, residue_i, residue_j, plot_frequencies=True,
                plot_type="heatmap", plot_out=None)


        header = html.H3("Pairwise AA Frequencies for residue pair {0} - {1}".format(residue_i, residue_j))
        graph_element = dcc.Graph( id='graph', figure=figure, style={'height': 800} )

        return html.Div([header, graph_element], style={'text-align': 'center'})



@app.callback(Output('tab-output-3', 'children'),
              [Input('tabs', 'value'),
               Input('protein_alignment', 'children'),
               Input('protein_pdb', 'children'),
               Input('protein_braw', 'children'),
               Input('contact_score_correction', 'value'),
               Input('sequence_separation', 'value'),
               Input('contact_threshold', 'value')
               ])
def display_tab_3(value, protein_alignment_json, protein_pdb_json, protein_braw_json, correction, seq_sep, contact_threshold):

    if value == 3 and protein_braw_json is not None:

        protein_braw = json.loads(protein_braw_json)
        figure = {}

        if 'x_pair' in protein_braw:

            L = u.find_dict_key('ncol', protein_braw['meta']['workflow'][0])
            braw_x_pair = np.array(protein_braw['x_pair']).reshape((L, L, 20, 20))

            alignment=None
            if protein_alignment_json is not None:
                protein_alignment = json.loads(protein_alignment_json)
                alignment = np.array(protein_alignment['alignment']).reshape((protein_alignment['N'], L))

            observed_distances = None
            if protein_pdb_json is not None:
                protein_pdb = json.loads(protein_pdb_json)
                observed_distances = np.array(protein_pdb['distance_map']).reshape((L,L))


            if correction == "ec" and alignment is not None:
                single_freq, pair_freq = au.calculate_frequencies(alignment, au.uniform_pseudocounts)
                mat = bu.compute_entropy_corrected_mat(braw_x_pair, single_freq, squared=False)
            elif correction == "apc":
                mat = bu.compute_l2norm_from_braw(braw_x_pair, True)
            else:
                mat = bu.compute_l2norm_from_braw(braw_x_pair, False)


            ### if alignment file is specified, compute gaps
            if alignment is not None:
                gaps_percentage_plot = aligncov.plot_percentage_gaps_per_position(alignment, plot_file=None)
            else:
                gaps_percentage_plot = None

            plot_matrix = pd.DataFrame()
            indices_upper_tri = np.triu_indices(L, seq_sep)

            #get residue-residue distance information from PDB
            if observed_distances is not None:
                plot_matrix['distance'] = observed_distances[indices_upper_tri]
                plot_matrix['contact'] = ((plot_matrix.distance < contact_threshold) * 1).tolist()

            # add scores
            plot_matrix['residue_i'] = indices_upper_tri[0] + 1
            plot_matrix['residue_j'] = indices_upper_tri[1] + 1
            plot_matrix['confidence'] = mat[indices_upper_tri]

            ### Plot Contact Map
            figure = plot.plot_contact_map_someScore_plotly(
                plot_matrix, "", seq_sep, gaps_percentage_plot, plot_file=None)



        header = html.H4("Contact Map with correction = {0} using sequence separation = {1} and contact threshold = {2}".format(correction, seq_sep, contact_threshold))
        graph_element = dcc.Graph( id='graph', figure=figure, style={'height': 600})

        return html.Div([header, graph_element], style={'text-align': 'center'})


@app.callback(Output('tab-output-4', 'children'),
              [Input('tabs', 'value'),
               Input('protein_braw', 'children'),
               Input('protein_pdb', 'children'),
               Input('protein_alignment', 'children'),
               Input('sequence_separation', 'value'),
               Input('contact_threshold', 'value')
               ])
def display_tab_4(value, protein_braw_json, protein_pdb_json, protein_alignment_json, seq_sep, contact_threshold):


    if value == 4 and protein_braw_json is not None and protein_pdb_json is not None:

        protein_braw = json.loads(protein_braw_json)
        protein_pdb = json.loads(protein_pdb_json)
        alignment = json.loads(protein_alignment_json)

        L = u.find_dict_key('ncol', protein_braw['meta']['workflow'][0])
        braw_x_pair = np.array(protein_braw['x_pair']).reshape((L, L, 20, 20))

        observed_distances = np.array(protein_pdb['distance_map']).reshape((L,L))

        dict_scores = {}
        for correction in ["apc", "no"]: #extend computation of corrections

            if correction == "ec" and alignment is not None:
                single_freq, pair_freq = au.calculate_frequencies(alignment, au.uniform_pseudocounts)
                mat = bu.compute_entropy_corrected_mat(braw_x_pair, single_freq, squared=False)
            elif correction == "apc":
                mat = bu.compute_l2norm_from_braw(braw_x_pair, True)
            else:
                mat = bu.compute_l2norm_from_braw(braw_x_pair, False)

            dict_scores['frobenius-' + correction] = mat


        figure = prec_plot.plot_precision_vs_rank(
            dict_scores, observed_distances, seq_sep, contact_threshold, "", plot_out=None)

        header = html.H3("Precision vs Rank ")
        graph_element = dcc.Graph( id='graph', figure=figure, style={'height': 600})
        return html.Div([header, graph_element], style={'text-align': 'center'})


@app.callback(Output('tab-output-5', 'children'),
              [Input('tabs', 'value'),
               Input('protein_braw', 'children'),
               Input('protein_alignment', 'children'),
               Input('res_i', 'value'),
               Input('res_j', 'value'),
               Input('coupling_matrix_correction', 'value')
               ])
def display_tab_5(value, protein_braw_json, protein_alignment_json, residue_i, residue_j, correction):



    if value == 5 and protein_braw_json is not None:
        protein_braw = json.loads(protein_braw_json)

        figure = {}

        # if len(path_dict['braw_file']) > 0:
            # L = protein_data_dict['L']
            # N = protein_data_dict['N']
            # lambda_w = protein_data_dict['lambda_w']
            # neff = protein_data_dict['neff']
            #
            # braw_xpair = np.array(protein_data_dict['x_pair']).reshape((L, L, 20, 20))
            # braw_xsingle = np.array(protein_data_dict['x_single']).reshape((L, 20))

            # if correction != "no":
            #     alignment = np.array(protein_data_dict['alignment']).reshape((N, L))
            #     single_freq, pair_freq = au.calculate_frequencies(alignment, au.uniform_pseudocounts)
            #
            #     if correction == "ec":
            #         ui, correction_for_braw, eta = bu.compute_correction_ij(
            #             single_freq, neff, lambda_w, braw_xpair, residue_i, residue_j, entropy=True, squared=True)
            #     elif correction == "cc":
            #         ui, correction_for_braw, eta = bu.compute_correction_ij(
            #             single_freq, neff, lambda_w, braw_xpair, residue_i, residue_j, entropy=False, squared=True)
            #
            #     braw_sq = braw_xpair[:,:, :20, :20]  * braw_xpair[:,:, :20, :20]
            #     braw_corrected = braw_sq - eta * correction_for_braw
            #
            #     figure = coupling_matrix_plot.plot_coupling_matrix_i_j(
            #         braw_corrected, ui, protein_name, residue_i, residue_j, eta, plot_out=None, plot_type="heatmap")
            #
            # else:
            #     figure = coupling_matrix_plot.plot_coupling_matrix_i_j(
            #         braw_xpair, braw_xsingle, protein_name, residue_i, residue_j, plot_out=None, plot_type="heatmap")

        header = html.H3("Coupling matrix for residue pair {0} - {1}".format(residue_i, residue_j))
        graph_element = dcc.Graph( id='graph', figure=figure, style={'height': 800} )
        return html.Div([header, graph_element], style={'text-align': 'center'})



# Run the Dash app
if __name__ == '__main__':
    app.server.run(debug=True, threaded=True)