# -*- coding: utf-8 -*-
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import flask
import os
import numpy as np
import base64
import glob
import json
import contact_prediction.utils.io_utils as io
import contact_prediction.utils.utils as u
import contact_prediction.utils.pdb_utils as pdb
import contact_prediction.utils.alignment_utils as au
import contact_prediction.utils.ccmraw as raw
import contact_prediction.plotting.plot_alignment_aminoacid_distribution as alignment_plot
#####import coupling_matrix_analysis.plot_coupling_matrix as coupling_matrix_plot
#####import plotting.plot_contact_map as contact_map_plot
#####import plotting.plot_pairwise_aa_freq as pairwise_aa_plot
####import utils.benchmark_utils as bu
#####import utils.plot_utils as plots

server = flask.Flask(__name__)
server.secret_key = os.environ.get('secret_key', 'secret')
app = dash.Dash(name = __name__, server = server)
app.config.supress_callback_exceptions = True

app.css.config.serve_locally = True
app.scripts.config.serve_locally = True

def_path_braw="/home/vorberg/work/data//benchmarkset_cathV4.1/contact_prediction/ccmpred-pll-centerv/braw/"
def_path_alignment="/home/vorberg/work/data//benchmarkset_cathV4.1/psicov/"
def_path_pdb="/home/vorberg/work/data//benchmarkset_cathV4.1/pdb_renum_combs/"


############################################################
# Define Layout
############################################################

app.layout = html.Div([
    ##Alignment Stats Top Right
    html.Div(id="alignment_stats",
             style={'position': 'absolute', 'left': '35%', 'top': '3%', 'width': '63%', 'height': '10%'}),

    ## tabs and Output Right
    html.Div([
        dcc.Tabs(
            tabs=[{'label': 'Alignment', 'value': 1},
                  {'label': 'Pairwise AA Freq', 'value': 2},
                  {'label': 'Contact Maps', 'value': 3},
                  {'label': 'Coupling Matrices', 'value': 4}],
            value=1,
            id='tabs'
        ),
        html.Div(id='tab-output-1',style={'display': 'block'}),
        html.Div(id='tab-output-2',style={'display': 'none'}),
        html.Div(id='tab-output-3',style={'display': 'none'}),
        html.Div(id='tab-output-4',style={'display': 'none'}),
    ], style={'position': 'absolute', 'left': '35%', 'top': '15%', 'width': '63%', 'height': '80%'}),


    ## Menu - left

    html.Div([

        html.Label("Upload a binary raw file:", style={'font-size': '16px'}),
        html.Br(),
        dcc.Upload(
            id='upload-alignment',
            children=[html.Button('Upload File')],
            multiple=False,
            style={
                'width': '100%',
                'height': '60px',
                'lineHeight': '60px',
                'borderWidth': '1px',
                'borderStyle': 'dashed',
                'borderRadius': '5px',
                'textAlign': 'center',
                'margin': '10px'
            }
        ),
        html.Br(),
        html.Br(),

        html.Label("Path to alignment files:", style={'font-size': '16px'}),
        html.Br(),
        dcc.Input(
            type='text',
            value=def_path_alignment,
            id='alignment_dir',
            size=70
        ),
        html.Br(),
        html.Br(),

        html.Label("Path to PDB files:", style={'font-size': '16px'}),
        html.Br(),
        dcc.Input(
            type='text',
            value=def_path_pdb,
            id='pdb_dir',
            size=70
        ),
        html.Br(),
        html.Br(),

        html.Label("Protein name:", style={'font-size': '16px'}),
        html.Br(),
        dcc.Input(
            type='text',
            value='1mkcA00',
            id="protein_name"
        ),
        html.Br(),
        html.Br(),

        html.Button('Submit', id='button',
                    style={'background-color': 'white', 'border': '2px solid #4CAF50', 'color' : 'black', 'font-size': '16px', 'padding': '15px 32px', 'border-radius': '4px'}),
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
                    style={'display': 'inline', 'width': '20'}
                ),
                html.Div(
                    children = [
                        html.Label("Residue Pair j:  ", style={'font-size': '16px'}),
                        dcc.Dropdown(
                            options=[{'label': str(i), 'value': str(i), 'disabled': 'True'} for i in range(2,3)],
                            id="res_j",
                            value="2")
                    ],
                    style={'display': 'inline', 'max-width': '20'}
                ),
                html.Br()
            ],
            style={'display': 'none'}
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
                    value='no'
                )
            ],
            style={'display': 'none'}
        ),


        html.Div(
            id='contact_map_options',
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
                    value='apc'
                ),
                html.Br(),
                html.Label("Sequence separation:  ", style={'font-size': '16px'}),
                dcc.Slider(
                    id='sequence_separation',
                    min=1,
                    max=15,
                    step=1,
                    value=6,
                    marks={i: str(i) for i in range(1,16)}
                ),
                html.Br(),
                html.Label("Contact Threshold:  ", style={'font-size': '16px'}),
                dcc.Slider(
                    id='contact_threshold',
                    min=4,
                    max=15,
                    step=1,
                    value=8,
                    marks={i: str(i) for i in range(4,16)}
                )
            ],
            style={'display': 'none'}
        )

    ], style={'position': 'absolute', 'left': '3%', 'top': '3%', 'width' : '30%'}),




    # html.Div([
    #     dcc.Graph(id='alignment_graph')
    # ], style={'position': 'absolute', 'left': '35%', 'top': '3%', 'width' : '63%', 'height': '40%'}),
    #
    # html.Div([
    #     dcc.Graph(id='pairwise_aa_freq_graph')
    # ], style={'position': 'absolute', 'left': '35%', 'top': '43%', 'width': '63%', 'height': '40%'}),

    # Hidden div inside the app that stores the intermediate value
    html.Div(id='protein_paths', style={'display': 'none'}),
    html.Div(id='protein_data', style={'display': 'none'})

])







############################################################
# Reactivity
############################################################


# @app.callback(Output('protein_paths', 'children'),
#               [Input('button', 'n_clicks')],
#               [State('alignment_dir', 'value'),
#                State('braw_dir', 'value'),
#                State('pdb_dir', 'value'),
#                State('protein_name', 'value')]
#               )
# def set_file_paths(n_clicks, alignment_dir, braw_dir, pdb_dir, protein_name):
#
#     path_dict = {
#         'alignment_file': glob.glob(alignment_dir + "/*" + protein_name + "*.psc")[0],
#         'braw_file': glob.glob(braw_dir + "/*" + protein_name + "*.braw.gz")[0],
#         'pdb_file': glob.glob(pdb_dir + "/*" + protein_name + "*.pdb")[0],
#         'protein_name': protein_name
#     }
#
#     return json.dumps(path_dict)



@app.callback(Output('protein_data', 'children'),
                [Input('button', 'n_clicks'),
                Input('upload-alignment', 'contents'),
                Input('upload-alignment', 'filename')]
              )
def load_alignment_data(n_clicks, alignment_contents_list, alignment_filename):

    protein_data = {}

    if alignment_contents_list is not None:
        print(alignment_filename)

        print(len(alignment_contents_list))

        print(alignment_contents_list)

        content_type, content_string = alignment_contents_list.split(',')
        decoded = base64.b64decode(content_string)

        print(decoded)

        r = base64.decodestring(content_string)

        print(r)

        ar_string = np.frombuffer(r, dtype=np.str)

        print(ar_string)

        alignment = np.array([[io.AMINO_INDICES[c] for c in x.strip()] for x in ar_string], dtype=np.uint8)


        protein_data['N'] = alignment.shape[0]
        protein_data['L'] = alignment.shape[1]
        protein_data['alignment'] = alignment.reshape(protein_data['N'] * protein_data['L']).tolist()



# @app.callback(Output('protein_data', 'children'),
#               [Input('protein_paths', 'children')]
#               )
# def load_data(protein_paths_json):
#
#     path_dict =  json.loads(protein_paths_json)
#     protein_data = {}
#
#     if len(path_dict['braw_file']) > 0:
#         braw = raw.parse_msgpack(path_dict['braw_file'])
#
#         protein_data['N'] = u.find_dict_key('nrow', braw.meta['workflow'][0])
#         protein_data['L'] = u.find_dict_key('ncol', braw.meta['workflow'][0])
#         protein_data['neff'] = u.find_dict_key('neff', braw.meta['workflow'][0])
#         protein_data['diversity'] = np.sqrt(protein_data['N']) / protein_data['L']
#         protein_data['lambda_w'] = u.find_dict_key('lambda_pair', braw.meta['workflow'][0])
#
#         protein_data['x_pair'] = braw.x_pair[:, :, :20, :20].reshape(protein_data['L'] * protein_data['L'] * 20 * 20).tolist()
#         protein_data['x_single'] = braw.x_single[:, :20].reshape(protein_data['L'] * 20).tolist()
#
#     if len(path_dict['alignment_file']) > 0:
#         alignment = io.read_alignment(path_dict['alignment_file'])
#         protein_data['alignment'] = alignment.reshape(protein_data['N'] * protein_data['L']).tolist()
#
#     if len(path_dict['pdb_file']) > 0:
#         distance_map = pdb.distance_map(path_dict['pdb_file'], L=protein_data['L'])
#         protein_data['distance_map'] = distance_map.reshape(protein_data['L'] * protein_data['L']).tolist()
#
#     return json.dumps(protein_data)


@app.callback(Output('res_i', 'options'), [Input('protein_data', 'children')])
def update_res_i(protein_data_json):
    protein_data=json.loads(protein_data_json)
    dropdown_options = [{'label': str(i), 'value': str(i)} for i in range(1, protein_data['L']-1)]
    return(dropdown_options)


@app.callback(Output('res_j', 'options'), [Input('res_i', 'value'), Input('res_i', 'options')])
def update_res_j(value, res_i_options):
    L = len(res_i_options)
    dropdown_options = [{'label': str(i), 'value': str(i)} for i in range(int(value)+1, int(L)+2)]
    return(dropdown_options)


@app.callback(Output('alignment_stats', 'children'), [Input('protein_data', 'children')])
def update_alignment_stats(protein_data_json):
    protein_data_dict=json.loads(protein_data_json)

    title1 = "L: " + str(protein_data_dict['L'])
    title2 = "N: " + str(protein_data_dict['N'])
    title3 = "Neff: " + str(np.round(protein_data_dict['neff'], decimals=3))
    title4 = "diversity: " + str(np.round(protein_data_dict['diversity'], decimals=3))

    header_1 = html.H3("Alignment Statistics")

    table = html.Table([
        html.Tr([
            html.Td("protein length", style={'padding': '5'}),
            html.Td("number of sequences", style={'padding': '5'}),
            html.Td("Neff", style={'padding': '5'}),
            html.Td("Diversity", style={ 'padding': '5'})
        ], style={'background': 'white', 'font-weight': 'bold'}),
        html.Tr([
            html.Td(protein_data_dict['L'], style={'padding': '5'}),
            html.Td(protein_data_dict['N'], style={'padding': '5'}),
            html.Td(np.round(protein_data_dict['neff'], decimals=3), style={ 'padding': '5'}),
            html.Td(np.round(protein_data_dict['diversity'], decimals=3), style={'padding': '5'})
        ], style={'background': 'white', 'font-weight': 'normal'})
    ], style={'border-collapse': 'collapse', 'margin-left': 'auto', 'margin-right': 'auto'})


    return html.Div([header_1, table], style={'text-align': 'center'})

############################################################
# Menu display according to Tab
############################################################


@app.callback(Output('contact_map_options', 'style'), [Input('tabs', 'value')])
def adjust_menu(value):

    if value == 1:
        return {'display': 'none'}
    elif value == 2:
        return {'display': 'none'}
    elif value == 3:
        return {'display': 'inline'}
    elif value == 4:
        return {'display': 'none'}

@app.callback(Output('residue_pair_ids', 'style'), [Input('tabs', 'value')])
def adjust_menu(value):

    if value == 1:
        return {'display': 'none'}
    elif value == 2:
        return {'display': 'inline'}
    elif value == 3:
        return {'display': 'none'}
    elif value == 4:
        return {'display': 'inline'}

@app.callback(Output('coupling_matrix_options', 'style'), [Input('tabs', 'value')])
def adjust_menu(value):

    if value == 1:
        return {'display': 'none'}
    elif value == 2:
        return {'display': 'none'}
    elif value == 3:
        return {'display': 'none'}
    elif value == 4:
        return {'display': 'inline'}


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
############################################################
# Tab Display
############################################################


@app.callback(Output('tab-output-1', 'children'),
              [Input('tabs', 'value'),
               Input('protein_paths', 'children')])
def display_tab_1(value, protein_paths_json):

    path_dict = json.loads(protein_paths_json)

    if value == 1:
        figure={}
        if len(path_dict['alignment_file']) > 0:
            #new version
            #figure = alignment_plot.plot_amino_acid_distribution_per_position(path_dict['alignment_file'], plot_file=None, freq=False)
            #old version
            figure = alignment_plot.plot_amino_acid_distribution_per_position(path_dict['alignment_file'], plot_file=None, freq=False)

        graph_element = dcc.Graph( id='graph', figure=figure, style={'height': 800} )
        return html.Div([graph_element], style={'text-align': 'center'})


@app.callback(Output('tab-output-2', 'children'),
              [Input('tabs', 'value'),
               Input('protein_paths', 'children'),
               Input('res_i', 'value'),
               Input('res_j', 'value')
               ])
def display_tab_2(value, protein_paths_json, residue_i, residue_j):

    path_dict = json.loads(protein_paths_json)

    if value == 2:
        figure = {}

        # if len(path_dict['alignment_file']) > 0:
        #     figure = pairwise_aa_plot.plot_aa_frequencies(
        #         path_dict['alignment_file'], residue_i, residue_j, plot_frequencies=True, plot_type="heatmap",
        #         plot_out=None)

        header = html.H3("Pairwise AA Frequencies for residue pair {0} - {1}".format(residue_i, residue_j))
        graph_element = dcc.Graph( id='graph', figure=figure, style={'height': 800} )

        return html.Div([header, graph_element], style={'text-align': 'center'})



@app.callback(Output('tab-output-3', 'children'),
              [Input('tabs', 'value'),
               Input('protein_paths', 'children'),
               Input('protein_data', 'children'),
               Input('contact_score_correction', 'value'),
               Input('sequence_separation', 'value'),
               Input('contact_threshold', 'value')
               ])
def display_tab_3(value, protein_paths_json, protein_data_json,  correction, seq_sep, contact_threshold):

    path_dict = json.loads(protein_paths_json)
    protein_name = path_dict['protein_name']
    protein_data_dict = json.loads(protein_data_json)

    if value == 3:
        figure = {}

        #if len(path_dict['braw_file']) > 0:
        #
        #     braw_xpair = np.array(protein_data_dict['x_pair']).reshape((L, L, 20, 20))
        #
        #     if correction == "ec" and len(path_dict['alignment_file']) > 0:
        #         alignment = protein_data_dict['alignment']
        #         single_freq, pair_freq = au.calculate_frequencies(alignment, au.uniform_pseudocounts)
        #         mat = bu.compute_entropy_corrected_mat(braw, single_freq, squared=False)
        #     elif correction == "apc":
        #         mat = bu.compute_l2norm_from_braw(braw, True)
        #     else:
        #         mat = bu.compute_l2norm_from_braw(braw, False)
        #
        #     alignment_file = None
        #     if len(path_dict['alignment_file']) > 0:
        #         alignment_file = path_dict['alignment_file']
        #
        #     pdb_file = None
        #     if len(path_dict['pdb_file']) > 0:
        #         pdb_file = path_dict['pdb_file']
        #
        #     title1 = " "
        #
        #
        #     figure = contact_map_plot.plot_contact_map(
        #         mat, seq_sep, contact_threshold, title,
        #         alignment_file=alignment_file,
        #         pdb_file=pdb_file,
        #         plot_file=None
        #     )


        header = html.H4("Contact Map with correction = {0} using sequence separation = {1} and contact threshold = {2}".format(correction, seq_sep, contact_threshold))
        graph_element = dcc.Graph( id='graph', figure=figure, style={'height': 800} )

        return html.Div([header, graph_element], style={'text-align': 'center'})

@app.callback(Output('tab-output-4', 'children'),
              [Input('tabs', 'value'),
               Input('protein_paths', 'children'),
               Input('protein_data', 'children'),
               Input('res_i', 'value'),
               Input('res_j', 'value'),
               Input('coupling_matrix_correction', 'value')
               ])
def display_tab_4(value, protein_paths_json, protein_data_json, residue_i, residue_j, correction):

    path_dict = json.loads(protein_paths_json)
    protein_name = path_dict['protein_name']
    protein_data_dict = json.loads(protein_data_json)

    if value == 4:
        figure = {}

        if len(path_dict['braw_file']) > 0:

            L = protein_data_dict['L']
            N = protein_data_dict['N']
            lambda_w = protein_data_dict['lambda_w']
            neff = protein_data_dict['neff']

            braw_xpair = np.array(protein_data_dict['x_pair']).reshape((L, L, 20, 20))
            braw_xsingle = np.array(protein_data_dict['x_single']).reshape((L, 20))

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