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
                  {'label': 'Amino Acid Frequencies', 'value': 2},
                  {'label': 'Contact Maps', 'value': 3},
                  {'label': 'Precision', 'value': 4},
                  {'label': 'Coupling Matrices', 'value': 5}
                  ],
            value=1,
            id='tabs'
        ),
        html.Div(id='tab-output',style={'display': 'block'}),
    ], style={'position': 'absolute', 'left': '25%', 'top': '15%', 'width': '63%', 'height': '80%'}),


    ## Menu - left

    html.Div([

        html.Label("Load example data:  ", style={'font-size': '16px'}),
        dcc.RadioItems(
            id='example_data',
            options=[
                {'label': 'none', 'value': 'no'},
                {'label': 'protein 1mkc_A_00', 'value': '1mkcA00'},
                {'label': 'protein 1c75_A_00', 'value': '1c75A00'}
            ],
            value='no',
            labelStyle={'display': 'block'}
        ),
        html.Br(),
        html.Label(" or upload your own data:  ", style={'font-size': '16px'}),
        dcc.Upload(
            id='upload-alignment',
            children=[
                html.Button('Upload Alignment File',
                            style={'background-color': 'white', 'border': '2px solid #4CAF50', 'color' : 'black',
                                   'font-size': '16px', 'padding': '5px 15px', 'border-radius': '4px'})
            ],
            multiple=False
        ),
        dcc.Upload(
            id='upload-pdb',
            children=[
                html.Button('Upload PDB File',
                            style={'background-color': 'white', 'border': '2px solid #4CAF50', 'color': 'black',
                                   'font-size': '16px', 'padding': '5px 15px', 'border-radius': '4px'})
            ],
            multiple=False
        ),
        dcc.Upload(
            id='upload-braw',
            children=[
                html.Button('Upload binary raw coupling File',
                            style={'background-color': 'white', 'border': '2px solid #4CAF50', 'color': 'black',
                                   'font-size': '16px', 'padding': '5px 15px', 'border-radius': '4px'})
            ],
            multiple=False
        ),

        html.Br(),
        html.Hr(),
        html.Br(),




        html.Div(
            id='menu-residue-pairs',
            children = [
                html.Label("Residue Pair i:  ", style={'font-size': '16px'}),
                dcc.Dropdown(
                    options=[{'label': str(i), 'value': str(i), 'disabled': 'True'} for i in range(1,2)],
                    id="res_i",
                    value="1"
                ),
                html.Label("Residue Pair j:  ", style={'font-size': '16px'}),
                dcc.Dropdown(
                    options=[{'label': str(i), 'value': str(i), 'disabled': 'True'} for i in range(2,3)],
                    id="res_j",
                    value="2"
                ),
                html.Br()
            ],
            style={'display': 'none', 'width': '10'}
        ),

        html.Div(
            id="menu-tab-3",
            children=[
                html.Label("Compute basic contact score from couplings as:  ", style={'font-size': '16px'}),
                dcc.RadioItems(
                    id='contact_score',
                    options=[
                        {'label': 'Frobenius Norm: ||w_ij||', 'value': 'frobenius'},
                        {'label': 'squared sum of couplings: ||w_ij||²', 'value': 'squared_sum'}
                    ],
                    value='frobenius',
                    labelStyle={'display': 'block'}
                ),
                html.Br(),
                html.Label("Apply correction to contact score:  ", style={'font-size': '16px'}),
                dcc.RadioItems(
                    id='contact_score_correction',
                    options=[
                        {'label': 'no correction', 'value': 'no'},
                        {'label': 'APC', 'value': 'apc'},
                        {'label': 'Entropy Correction', 'value': 'ec'},
                        {'label': 'Count Correction', 'value': 'cc'},
                        {'label': 'Pair Weighted Entropy Correction', 'value': 'pw-ec'},
                    ],
                    value='apc',
                    labelStyle={'display': 'block'}
                ),
                html.Br(),
                dcc.Checklist(
                    id="correction_only",
                    options=[
                        {'label': 'plot only correction', 'value': '1'}
                    ],
                    values=[],
                    labelStyle={'display': 'inline-block'}
                ),
                # html.Hr(),
                # html.Div("Entropy and Count Correction are defined either as:"),
                # html.Div(" η• √(sum_a,b u_ia•u_jb) when using Frobenius Norm Score or"),
                # html.Div(" η• sum_a,b u_ia•u_jb when using summed squares Score"),
                # html.Div("with u_ia = sqrt(neff)/lambda_w q_ia * log(qia) for Entropy Correction"),
                # html.Div("  or u_ia = sqrt(neff)/lambda_w q_ia * (1 - qia) for Count Correction"),
                # html.Br(),
                # html.Div("Pair Weighted Entropy Correction:"),
                # html.Div("sum_a,b beta_ab (w_ijab² - η•u_ia•u_jb)"),
                # html.Hr(),
                html.Br(),

            ],
            style={'display': 'none'}
        ),

        html.Div(
            id="menu-tab-5",
            children = [
                html.Br(),
                html.Label("Apply local correction to coupling values:  ", style={'font-size': '16px'}),
                dcc.RadioItems(
                    id='coupling_matrix_correction',
                    options=[
                        {'label': 'no correction', 'value': 'couplings with no correction'},
                        {'label': 'entropy corrected squared couplings', 'value': 'entropy corrected squared couplings'},
                        {'label': 'count corrected squared couplings', 'value': 'count corrected squared couplings'},
                        {'label': 'pair weighted, entropy corrected squared couplings', 'value': 'pair weighted, entropy corrected squared couplings'},
                        {'label': 'only entropy correction (no square root)', 'value': 'entropy correction (no square root)'},
                        {'label': 'only count correction (no square root)', 'value': 'count correction (no square root)'},
                        {'label': 'only pair weights', 'value': 'only pair weights'}
                    ],
                    value='couplings with no correction',
                    labelStyle={'display': 'block'}
                ),
            ],
            style={'display': 'none'}
        ),

        html.Div(
            id="seq-sep-contact-thresh",
            children=[
                html.Label("Sequence separation:  ", style={'font-size': '16px'}),
                dcc.Slider(
                    id='sequence_separation',
                    min=2,
                    max=12,
                    step=2,
                    value=6,
                    marks={i: str(i) for i in range(2, 12 + 1, 2)}
                ),
                html.Br(),
                html.Label("Contact Threshold:  ", style={'font-size': '16px'}),
                dcc.Slider(
                    id='contact_threshold',
                    min=4,
                    max=14,
                    step=2,
                    value=8,
                    marks={i: str(i) for i in range(4, 14 + 1, 2)}
                )
            ],
            style={'display': 'none'}
        ),

        html.Br(),
        html.Button(
            'Go',
            id='go',
            style={'background-color': 'white', 'border': '2px solid #4CAF50', 'color': 'black', 'font-size': '16px',
                   'padding': '10px 20px', 'border-radius': '4px'}
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
              [Input('upload-alignment', 'contents'), Input('upload-alignment', 'filename'), Input('example_data', 'value')])
def load_alignment_data(alignment_contents_list, alignment_name, example_protein):

    protein_alignment_dict = {}

    alignment = None
    if example_protein != "no":

        alignment = io.read_alignment('./example_data/'+example_protein+'.filt.psc')

        protein_alignment_dict['protein_name'] = example_protein
        protein_alignment_dict['alignment_filename'] = './'+example_protein+'.filt.psc'

    elif alignment_contents_list is not None:

        protein_alignment_dict['protein_name'] = alignment_name.split(".")[0]
        protein_alignment_dict['alignment_filename'] = alignment_name

        content_type, content_string = alignment_contents_list.split(',')
        decoded_string = base64.decodestring(content_string)
        decoded_split_str = decoded_string.split("\n")

        alignment = np.array([[io.AMINO_INDICES[c] for c in x.strip()] for x in decoded_split_str[:-1]], dtype=np.uint8)

    if alignment is not None:

        N = alignment.shape[0]
        L = alignment.shape[1]
        protein_alignment_dict['N'] = N
        protein_alignment_dict['L'] = L
        protein_alignment_dict['alignment'] = alignment.reshape(N * L).tolist()

        #compute amino acid frequencies incl sequence weighting and pseudocounts
        single_freq, pair_freq = au.calculate_frequencies(alignment, au.uniform_pseudocounts)
        protein_alignment_dict['single_freq'] = single_freq[:, :20].reshape(L * 20).tolist()
        protein_alignment_dict['pair_freq'] = pair_freq[:, :, :20, :20].reshape(L * L * 20 * 20).tolist()

        #also compute the counts without pseudo-counts
        single_counts, pairwise_counts = au.compute_counts(alignment, compute_weights=True)
        protein_alignment_dict['single_counts'] = single_counts[:, :20].reshape(L * 20).tolist()
        protein_alignment_dict['pair_counts'] = pairwise_counts[:, :, :20, :20].reshape(L * L * 20 * 20).tolist()

    return json.dumps(protein_alignment_dict)


@app.callback(Output('protein_pdb', 'children'),
              [
                  Input('upload-pdb', 'contents'),
                  Input('upload-pdb', 'filename'),
                  Input('example_data', 'value'),
                  Input('protein_alignment', 'children')
              ]
              )
def load_pdb_data(pdb_contents_list, pdb_name, example_protein, protein_alignment_json):

    protein_pdb_dict = {}

    #empty dict and None evaluates to False
    if protein_alignment_json:

        protein_alignment_dict = json.loads(protein_alignment_json)

        if 'L' in protein_alignment_dict:
            L = protein_alignment_dict['L']

            if example_protein != "no":

                protein_pdb_dict['protein_name'] = example_protein
                protein_pdb_dict['pdb_filename'] = './example_data/'+example_protein+'.pdb'

            elif pdb_contents_list is not None:

                protein_pdb_dict['protein_name'] = pdb_name.split(".")[0]
                protein_pdb_dict['pdb_filename'] = './tmp.pdb'

                content_type, content_string = pdb_contents_list.split(',')
                decoded_string = base64.decodestring(content_string)

                f = open(protein_pdb_dict['pdb_filename'], 'w')
                f.write(decoded_string)
                f.close()

            if 'pdb_filename' in protein_pdb_dict:

                distance_map = pdb.distance_map(protein_pdb_dict['pdb_filename'], L=L, distance_definition="Cb")
                protein_pdb_dict['distance_map'] = distance_map.reshape(L * L).tolist()

    return json.dumps(protein_pdb_dict)


@app.callback(Output('protein_braw', 'children'),
              [
                Input('upload-braw', 'contents'),
                Input('upload-braw', 'filename'),
                Input('example_data', 'value')
              ]
              )
def load_braw_data(braw_contents_list, braw_name, example_protein):


    protein_braw_dict = {}

    if example_protein != "no":

        protein_braw_dict['protein_name'] = example_protein
        protein_braw_dict['braw_filename'] = './example_data/'+example_protein+'.filt.braw.gz'

    elif braw_contents_list is not None:

        protein_braw_dict['protein_name'] = braw_name.split(".")[0]
        protein_braw_dict['braw_filename'] = './tmp.braw.gz'

        content_type, content_string = braw_contents_list.split(',')
        decoded_string = base64.decodestring(content_string)

        f = open(protein_braw_dict['braw_filename'], 'w')
        f.write(decoded_string)
        f.close()

    if 'braw_filename' in protein_braw_dict:

        braw = raw.parse_msgpack(protein_braw_dict['braw_filename'])
        L = braw.ncol

        protein_braw_dict['meta'] = braw.meta

        protein_braw_dict['x_single'] = braw.x_single[:, :20].reshape(L * 20).tolist()
        protein_braw_dict['x_pair'] = braw.x_pair[:, :, :20, :20].reshape(L * L * 20 * 20).tolist()

    return json.dumps(protein_braw_dict)


############################################################
# Reactivity - Statistics
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

@app.callback(Output('menu-residue-pairs', 'children'), [Input('protein_alignment', 'children')])
def update_residue_pairs(protein_alignment_json):

    protein_alignment_dict=json.loads(protein_alignment_json)
    L = 3
    if 'L' in protein_alignment_dict:
        L = protein_alignment_dict['L']

    children = [
        html.Label("Residue Pair i:  ", style={'font-size': '16px'}),
        dcc.Dropdown(
            options=[{'label': str(i), 'value': str(i)} for i in range(1, L)],
            id="res_i",
            value="1"
        ),
        html.Label("Residue Pair j:  ", style={'font-size': '16px'}),
        dcc.Dropdown(
            options=[{'label': str(i), 'value': str(i)} for i in range(2, L+1)],
            id="res_j",
            value=str(L)
        )
    ]

    return children



@app.callback(Output('menu-residue-pairs', 'style'), [Input('tabs', 'value')])
def adjust_menu_2(value):

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

@app.callback(Output('menu-tab-3', 'style'), [Input('tabs', 'value')])
def adjust_menu_3(value):

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

@app.callback(Output('seq-sep-contact-thresh', 'style'), [Input('tabs', 'value')])
def adjust_menu_4(value):

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

@app.callback(Output('menu-tab-5', 'style'), [Input('tabs', 'value')])
def adjust_menu_5(value):

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
# Tab Display
############################################################

@app.callback(Output('tab-output', 'children'),
              [Input('tabs', 'value'),
               Input('go', 'n_clicks')],
              [State('protein_alignment', 'children'),
               State('protein_pdb', 'children'),
               State('protein_braw', 'children'),
               State('res_i', 'value'),
               State('res_j', 'value'),
               State('contact_score', 'value'),
               State('contact_score_correction', 'value'),
               State('correction_only', 'values'),
               State('sequence_separation', 'value'),
               State('contact_threshold', 'value'),
               State('coupling_matrix_correction', 'value')
               ])
def display_tab_(
        value, n_clicks,
        protein_alignment_json, protein_pdb_json, protein_braw_json,
        residue_i_str, residue_j_str, contact_score,  correction, plot_correction_only,
        seq_sep, contact_threshold, coupling_matrix_correction):

    if value == 1:
        if protein_alignment_json:

            protein_alignment_dict = json.loads(protein_alignment_json)

            figure={}
            h2=""

            if 'L' in protein_alignment_dict:
                L = protein_alignment_dict['L']
                single_counts = np.array(protein_alignment_dict['single_counts']).reshape((L,20))

                figure = alignment_plot.plot_amino_acid_distribution_per_position(single_counts, "", plot_file=None, freq=False)
                h2=html.H2("Distribution of Amino Acids per position in alignment")

            graph_element = dcc.Graph( id='graph', figure=figure, style={'height': 800} )
            return html.Div([h2, graph_element], style={'text-align': 'center'})
        else:
            return html.Div(
                html.H3("You need to load an alingmnment for this analysis!"),
                style={'text-align': 'center'}
            )

    elif value == 2:
        if protein_alignment_json:

            protein_alignment_dict = json.loads(protein_alignment_json)
            figure = {}

            if 'alignment' in protein_alignment_dict:
                L = protein_alignment_dict['L']
                single_counts = np.array(protein_alignment_dict['single_counts']).reshape((L, 20))
                pairwise_counts = np.array(protein_alignment_dict['pair_counts']).reshape((L,L,20,20))

                protein_name = protein_alignment_dict['protein_name']

                figure = pairwise_aa_plot.plot_aa_frequencies(
                    single_counts, pairwise_counts, protein_name, int(residue_i_str), int(residue_j_str), plot_frequencies=True,
                    plot_type="heatmap", plot_out=None)


            header = html.H3("Single and Pairwise Amino Acid Frequencies for Residue Pair {0} - {1}".format(residue_i_str, residue_j_str))
            graph_element = dcc.Graph( id='graph', figure=figure, style={'height': 800} )

            return html.Div([header, graph_element], style={'text-align': 'center'})
        else:
            return html.Div(
                html.H3("You need to load an alingmnment for this analysis!"),
                style={'text-align': 'center'}
            )

    elif value == 3:
        if protein_braw_json and protein_alignment_json and protein_pdb_json:

            protein_braw = json.loads(protein_braw_json)
            protein_alignment = json.loads(protein_alignment_json)
            protein_pdb = json.loads(protein_pdb_json)

            figure = {}

            if 'x_pair' in protein_braw:

                print(protein_braw['meta']['workflow'][0])
                L = u.find_dict_key('ncol', protein_braw['meta']['workflow'][0])
                N = u.find_dict_key('nrow', protein_braw['meta']['workflow'][0])
                lambda_w = u.find_dict_key('lambda_pair', protein_braw['meta']['workflow'][0])
                neff = u.find_dict_key('neff', protein_braw['meta']['workflow'][0])
                braw_x_pair = np.array(protein_braw['x_pair']).reshape((L, L, 20, 20))


                single_freq = np.array(protein_alignment['single_freq']).reshape((L,20))
                alignment = np.array(protein_alignment['alignment']).reshape((N, L))
                observed_distances = np.array(protein_pdb['distance_map']).reshape((L,L))

                ### if alignment file is specified, compute gaps
                gaps_percentage_plot = aligncov.plot_percentage_gaps_per_position(alignment, plot_file=None)

                print(contact_score)
                print(correction)
                if contact_score == "frobenius":
                    if correction == "ec":
                        mat = bu.compute_corrected_mat(braw_x_pair, single_freq, neff, lambda_w, entropy=True, squared=False)
                    elif correction == "cc":
                        mat = bu.compute_corrected_mat(braw_x_pair, single_freq, neff, lambda_w, entropy=False, squared=False)
                    elif correction == "pw-ec":
                        return html.Div(html.H3("pair weights correction only applicable to squared sum score (weights have been optimized with squared sum score)!"), style={"text-align": "center"})
                    elif correction == "apc":
                        mat = bu.compute_l2norm_from_braw(braw_x_pair, apc=True, squared=False)
                    else:
                        mat = bu.compute_l2norm_from_braw(braw_x_pair, apc=False, squared=False)
                elif contact_score == 'squared_sum':
                    if correction == "ec":
                        mat = bu.compute_corrected_mat(braw_x_pair, single_freq, neff, lambda_w, entropy=True, squared=True)
                    elif correction == "cc":
                        mat = bu.compute_corrected_mat(braw_x_pair, single_freq, neff, lambda_w, entropy=False, squared=True)
                    elif correction == "pw-ec":
                        beta = np.loadtxt("./example_data/pair_weights_20000_balance5_contactthr8_noncontactthr20_diversitythr0.3_regcoeff10.txt")
                        uij, eta = bu.compute_correction(single_freq, neff, lambda_w, braw_x_pair, entropy=True, squared=True)
                        braw_sq = braw_x_pair[:, :, :20, :20] * braw_x_pair[:, :, :20, :20]
                        couplings_corrected = braw_sq - eta * uij
                        mat = np.sum(beta[np.newaxis, np.newaxis, :, :] * couplings_corrected[:,:,:20,:20], axis=(3,2))
                    elif correction == "apc":
                        mat = bu.compute_l2norm_from_braw(braw_x_pair, apc=True, squared=True)
                    else:
                        mat = bu.compute_l2norm_from_braw(braw_x_pair, apc=False, squared=True)

                print(plot_correction_only)
                if len(plot_correction_only) > 0:
                    if contact_score == "frobenius" and correction == "ec":
                        uij, scaling_factor_eta = bu.compute_correction(single_freq, neff, lambda_w, braw_x_pair, entropy=True, squared=False)
                        mat = scaling_factor_eta * np.sqrt(np.sum(uij, axis=(3, 2)))
                    elif contact_score == "squared_sum" and correction == "ec":
                        uij, scaling_factor_eta = bu.compute_correction(single_freq, neff, lambda_w, braw_x_pair, entropy=True, squared=True)
                        mat = scaling_factor_eta * np.sum(uij, axis=(3, 2))
                    elif contact_score == "frobenius" and correction == "cc":
                        uij, scaling_factor_eta = bu.compute_correction(single_freq, neff, lambda_w, braw_x_pair, entropy=False, squared=False)
                        mat = scaling_factor_eta * np.sqrt(np.sum(uij, axis=(3, 2)))
                    elif contact_score == "squared_sum" and correction == "cc":
                        uij, scaling_factor_eta = bu.compute_correction(single_freq, neff, lambda_w, braw_x_pair, entropy=False, squared=True)
                        mat = scaling_factor_eta * np.sum(uij, axis=(3, 2))
                    elif contact_score == "frobenius" and correction == "apc":
                        cmat = bu.compute_l2norm_from_braw(braw_x_pair, apc=False, squared=False)
                        mean = np.mean(cmat, axis=0)
                        mat = mean[:, np.newaxis] * mean[np.newaxis, :] / np.mean(cmat)
                    elif contact_score == "squared_sum" and correction == "apc":
                        cmat = bu.compute_l2norm_from_braw(braw_x_pair, apc=False, squared=True)
                        mean = np.mean(cmat, axis=0)
                        mat = mean[:, np.newaxis] * mean[np.newaxis, :] / np.mean(cmat)
                    elif correction == "pw-ec":
                        return html.Div(html.H3("Not applicable."), style={"text-align": "center"})
                    else:
                        return html.Div(html.H3("No correction selected."), style={"text-align": "center"})


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



            header = html.H3("Contact Map using contact score = {0} and correction = {1}".format(contact_score, correction))
            subheader = html.H3("with sequence separation = {0} and contact threshold = {1}".format(seq_sep, contact_threshold))
            graph_element = dcc.Graph( id='graph', figure=figure, style={'height': 700})

            return html.Div([header, subheader, graph_element], style={'text-align': 'center'})
        else:
            return html.Div(
                [
                    html.H3(
                        "You need to load an alignment file, a PDB file and a binary raw coupling file for this analysis!"
                    )
                ],
                style={'text-align': 'center'}
            )

    elif value == 4:
        if protein_braw_json and protein_pdb_json and protein_alignment_json:

            protein_braw = json.loads(protein_braw_json)
            L = u.find_dict_key('ncol', protein_braw['meta']['workflow'][0])
            neff = u.find_dict_key('neff', protein_braw['meta']['workflow'][0])
            lambda_w = u.find_dict_key('lambda_pair', protein_braw['meta']['workflow'][0])
            braw_x_pair = np.array(protein_braw['x_pair']).reshape((L, L, 20, 20))

            protein_pdb = json.loads(protein_pdb_json)
            observed_distances = np.array(protein_pdb['distance_map']).reshape((L,L))

            protein_alignment = json.loads(protein_alignment_json)
            single_freq = np.array(protein_alignment['single_freq']).reshape((L,20))


            dict_scores = {}
            ordered_methods = []
            for contact_score in ["frobenius", "squared sum"]:
                for correction in ["apc", "no", "entropy correction", "count correction", "pair weighted entropy correction"]: #extend computation of corrections

                    if contact_score == "frobenius":
                        if correction == "entropy correction":
                            mat = bu.compute_corrected_mat(
                                braw_x_pair, single_freq, neff, lambda_w, entropy=True, squared=False)
                            dict_scores[contact_score + " - " + correction] = mat
                            ordered_methods.append(contact_score + " - " + correction)
                        elif correction == "count correction":
                            mat = bu.compute_corrected_mat(
                                braw_x_pair, single_freq, neff, lambda_w, entropy=False, squared=False)
                            dict_scores[contact_score + " - " + correction] = mat
                            ordered_methods.append(contact_score + " - " + correction)
                        elif correction == "apc":
                            mat = bu.compute_l2norm_from_braw(braw_x_pair, apc=True, squared=False)
                            dict_scores[contact_score + " - " + correction] = mat
                            ordered_methods.append(contact_score + " - " + correction)
                        elif correction == "no":
                            mat = bu.compute_l2norm_from_braw(braw_x_pair, apc=False, squared=False)
                            dict_scores[contact_score + " - " + correction] = mat
                            ordered_methods.append(contact_score + " - " + correction)
                    elif contact_score == 'squared sum':
                        if correction == "entropy correction":
                            mat = bu.compute_corrected_mat(
                                braw_x_pair, single_freq, neff, lambda_w, entropy=True, squared=True)
                            dict_scores[contact_score + " - " + correction] = mat
                            ordered_methods.append(contact_score + " - " + correction)
                        elif correction == "count correction":
                            mat = bu.compute_corrected_mat(
                                braw_x_pair, single_freq, neff, lambda_w, entropy=False, squared=True)
                            dict_scores[contact_score + " - " + correction] = mat
                            ordered_methods.append(contact_score + " - " + correction)
                        elif correction == "pair weighted entropy correction":
                            beta = np.loadtxt("./example_data/pair_weights_20000_balance5_contactthr8_noncontactthr20_diversitythr0.3_regcoeff10.txt")
                            uij, eta = bu.compute_correction(
                                single_freq, neff, lambda_w, braw_x_pair, entropy=True, squared=True)
                            braw_sq = braw_x_pair[:, :, :20, :20] * braw_x_pair[:, :, :20, :20]
                            couplings_corrected = braw_sq - eta * uij
                            mat = np.sum(
                                beta[np.newaxis, np.newaxis, :, :] * couplings_corrected[:, :, :20, :20],
                                axis=(3, 2)
                            )
                            dict_scores[contact_score + " - " + correction] = mat
                            ordered_methods.append(contact_score + " - " + correction)

            figure = prec_plot.plot_precision_vs_rank(
                dict_scores, observed_distances, seq_sep, contact_threshold, "", ordered_methods, plot_out=None)


            header = html.H3("Precision vs Top Ranked Contact Predictions "
                             "(sequence separation = {0} and contact threshold = {1})".format(seq_sep, contact_threshold))
            graph_element = dcc.Graph( id='graph', figure=figure, style={'height': 700})
            return html.Div([header, graph_element], style={'text-align': 'center'})

        else:
            return html.Div(
                [
                    html.H3(
                        "You need to load an alignment file, a PDB file and a binary raw coupling file for this analysis!"
                    )
                ],
                style={'text-align': 'center'}
            )

    elif value == 5:
        if protein_braw_json is not None and protein_alignment_json is not None:

            protein_braw = json.loads(protein_braw_json)
            protein_alignment = json.loads(protein_alignment_json)


            L = protein_alignment['L']
            N = protein_alignment['N']
            lambda_w = u.find_dict_key('lambda_pair', protein_braw['meta']['workflow'][0])
            neff = u.find_dict_key('neff', protein_braw['meta']['workflow'][0])

            braw_x_pair = np.array(protein_braw['x_pair']).reshape((L, L, 20, 20))
            braw_xsingle = np.array(protein_braw['x_single']).reshape((L, 20))

            residue_i = int(residue_i_str)
            residue_j = int(residue_j_str)

            single_terms_i = braw_xsingle[residue_i - 1][:20]
            single_terms_j = braw_xsingle[residue_j - 1][:20]


            print(coupling_matrix_correction)
            if coupling_matrix_correction != "couplings with no correction":
                alignment = np.array(protein_alignment['alignment'], dtype='uint8').reshape((N, L))
                single_freq, pair_freq = au.calculate_frequencies(alignment, au.uniform_pseudocounts)

                if coupling_matrix_correction == 'entropy corrected squared couplings':
                    ui, correction_for_braw_ij, eta = bu.compute_correction_ij(
                        single_freq, neff, lambda_w, braw_x_pair, residue_i, residue_j, entropy=True, squared=True)
                    braw_sq = braw_x_pair[:, :, :20, :20] * braw_x_pair[:, :, :20, :20]
                    couplings_corrected = braw_sq[residue_i - 1, residue_j - 1, :20, :20] - eta * correction_for_braw_ij

                    print(couplings_corrected[io.AMINO_INDICES['C'], io.AMINO_INDICES['C']])

                    header = html.H3(
                        "Single Potentials and corrected Coupling Matrix ({0}) for Residue Pair {1} - {2}".format(
                            coupling_matrix_correction, residue_i, residue_j))


                    figure = plot.plot_coupling_matrix(
                        couplings_corrected, single_terms_i, single_terms_j,
                        residue_i, residue_j, "", "corrected coupling strength", 'diverging', type="heatmap", plot_file=None
                    )

                elif coupling_matrix_correction == 'entropy correction (no square root)':
                    ui, correction_for_braw_ij, eta = bu.compute_correction_ij(
                        single_freq, neff, lambda_w, braw_x_pair, residue_i, residue_j, entropy=True, squared=True)
                    couplings_corrected = eta * correction_for_braw_ij

                    print(couplings_corrected[io.AMINO_INDICES['C'], io.AMINO_INDICES['C']])

                    header = html.H3(
                        "Single Correction Values (u_i) and Correction Matrix ({0}) for Residue Pair {1} - {2}".format(
                            coupling_matrix_correction, residue_i, residue_j)
                    )
                    figure = plot.plot_coupling_matrix(
                        couplings_corrected, ui[residue_i - 1], ui[residue_j - 1],
                        residue_i, residue_j, "", "correction strength", 'continuous', type="heatmap", plot_file=None
                    )

                elif coupling_matrix_correction == 'count corrected squared couplings':
                    ui, correction_for_braw_ij, eta = bu.compute_correction_ij(
                        single_freq, neff, lambda_w, braw_x_pair, residue_i, residue_j, entropy=False, squared=True)
                    braw_sq = braw_x_pair[:, :, :20, :20] * braw_x_pair[:, :, :20, :20]
                    couplings_corrected = braw_sq[residue_i - 1, residue_j - 1, :20, :20] - eta * correction_for_braw_ij

                    header = html.H3(
                        "Single Potentials and corrected Coupling Matrix ({0}) for Residue Pair {1} - {2}".format(
                            coupling_matrix_correction, residue_i, residue_j)
                    )

                    figure = plot.plot_coupling_matrix(
                        couplings_corrected, single_terms_i, single_terms_j,
                        residue_i, residue_j, "", "corrected coupling strength", 'diverging', type="heatmap", plot_file=None
                    )

                elif coupling_matrix_correction == 'count correction (no square root)':
                    ui, correction_for_braw_ij, eta = bu.compute_correction_ij(
                        single_freq, neff, lambda_w, braw_x_pair, residue_i, residue_j, entropy=False, squared=True)
                    couplings_corrected = eta * correction_for_braw_ij

                    header = html.H3(
                        "Single Correction Values (u_i) and Correction Matrix ({0}) for Residue Pair {1} - {2}".format(
                            coupling_matrix_correction, residue_i, residue_j)
                    )

                    figure = plot.plot_coupling_matrix(
                        couplings_corrected, ui[residue_i - 1], ui[residue_j - 1],
                        residue_i, residue_j, "", "correction strength", 'continuous', type="heatmap", plot_file=None
                    )
                elif coupling_matrix_correction == 'pair weighted, entropy corrected squared couplings':
                    beta = np.loadtxt(
                        "./example_data/pair_weights_20000_balance5_contactthr8_noncontactthr20_diversitythr0.3_regcoeff10.txt")

                    ui, correction_for_braw_ij, eta = bu.compute_correction_ij(
                        single_freq, neff, lambda_w, braw_x_pair, residue_i, residue_j, entropy=True, squared=True)
                    braw_sq = braw_x_pair[:, :, :20, :20] * braw_x_pair[:, :, :20, :20]
                    couplings_corrected = braw_sq[residue_i - 1, residue_j - 1, :20, :20] - eta * correction_for_braw_ij
                    pair_weights_couplings_corrected = beta * couplings_corrected

                    print(beta[io.AMINO_INDICES['C'], io.AMINO_INDICES['C']])
                    print(couplings_corrected[io.AMINO_INDICES['C'], io.AMINO_INDICES['C']])

                    header = html.H3(
                        "Single Potentials and corrected Coupling Matrix ({0}) for Residue Pair {1} - {2}".format(
                            coupling_matrix_correction, residue_i, residue_j)
                    )

                    figure = plot.plot_coupling_matrix(
                        pair_weights_couplings_corrected, single_terms_i, single_terms_j,
                        residue_i, residue_j, "", "correction strength", 'diverging', type="heatmap", plot_file=None
                    )
                elif coupling_matrix_correction == 'only pair weights':
                    beta = np.loadtxt(
                        "./example_data/pair_weights_20000_balance5_contactthr8_noncontactthr20_diversitythr0.3_regcoeff10.txt")

                    header = html.H3(
                        "Matrix of Pair Weights".format(
                            residue_i, residue_j)
                    )

                    figure = plot.plot_coupling_matrix(
                        beta, np.zeros(20), np.zeros(20),
                        residue_i, residue_j, "", "pair weights", 'diverging', type="heatmap", plot_file=None
                    )

            else:


                couplings = braw_x_pair[residue_i - 1, residue_j - 1, :20, :20]
                figure = plot.plot_coupling_matrix(
                    couplings, single_terms_i, single_terms_j,
                    residue_i, residue_j, "", "coupling strength",'diverging', type="heatmap", plot_file=None
                )

                header = html.H3("Single Potentials and Coupling Matrix for Residue Pair {0} - {1}".format(residue_i, residue_j))

        graph_element = dcc.Graph( id='graph', figure=figure, style={'height': 700} )
        return html.Div([header, graph_element], style={'text-align': 'center', 'display': 'inline'})


# Run the Dash app
if __name__ == '__main__':
    app.server.run(debug=True, threaded=True)