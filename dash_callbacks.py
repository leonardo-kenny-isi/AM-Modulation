from dash import Dash, dcc, html, Input, Output, State
import pandas as pd
import numpy as np

app = Dash(__name__)

app.layout = html.Div([
    html.H6("Change the value in the text box to see callbacks in action!"),
    html.Div([
        "Input: ",
        dcc.Input(id='my-input', value='initial value', type='text'),
        html.Button(id='my-button', n_clicks=0, children='Submit'),
    ]),
    html.Br(),
    html.Div(id='my-output'),
    
    html.Br(),
    html.Div(id='other-output'),
    html.Div(id='other-output2'),
    
    dcc.Store(id='data')
])


@app.callback(
    Output(component_id='my-output', component_property='children'),
    Output(component_id='data', component_property='data'),
    Input(component_id='my-button', component_property='n_clicks'),
    State(component_id='my-input', component_property='value')
)
def update_output_div(button,input_value):
    df_test = pd.DataFrame(columns=["message"])
    df_test['message'] = np.array(input_value.split(',')).astype(np.float32)
    df_test['raw'] = np.array(input_value.split(','))
    return f'Output: {input_value}',df_test.to_json(orient='split',date_format='iso')

@app.callback(
    Output(component_id='other-output', component_property='children'),
    Output(component_id='other-output2', component_property='children'),
    Input(component_id='data', component_property='data'),
)
def update_output_div(input_value):
    df_test = pd.read_json(input_value,orient='split')
    df_test['message'] = np.array(df_test['message']).astype(np.float32)
    return f'Output 2: {df_test["message"]}'

if __name__ == '__main__':
    app.run_server(debug=True)
