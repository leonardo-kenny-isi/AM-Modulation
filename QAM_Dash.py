from dash import Dash, html, dash_table, dcc, callback, Output, Input, State
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import wave
import scipy
from scipy.fft import rfft as fft, rfftfreq as fftfreq

FSH = 8_820_000     # sampling frequency high frequency(Hz)
FSL = 44_100        # sampling frequency low frequency(Hz)
PERIOD = 0.1
# Initialize the app - incorporate css
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div([
    html.Div(className='row', children='AM Modulation - QAM',
             style={'textAlign': 'center', 'color': 'blue', 'fontSize': 30}),

    html.Div(className='row', children=[
        html.Div(className='six columns', children=[
                "M1 Frequency: ", dcc.Input(id='m1_f', value=300, type='text'),
                " M1_Factor: ", dcc.Input(id='m1_factor', value=0.2, type='number'),
                html.Button(id='submit_state_m1', n_clicks=0, children='Update M1')],
                style={'textAlign': 'center'}),
        html.Div(className='six columns',children=[
                "M2 Frequency: ", dcc.Input(id='m2_f', value=600, type='text'),
                " M2_Factor: ", dcc.Input(id='m2_factor', value=0.5, type='number'),
                html.Button(id='submit_state_m2', n_clicks=0, children='Update M2')],
                style={'textAlign': 'center'}),
    ]),

    html.Div(className='row', children=[
        html.Div(className='six columns',children=[dcc.Graph(figure={}, id='message1_fig')]),
        html.Div(className='six columns',children=[dcc.Graph(figure={}, id='message2_fig')]),
    ]),
    html.Br(),

    html.Div(["Carrier Frequency: ", dcc.Input(id='carrier_freq', value=800_000, type='number'),
              " Carrier Phase Deviation: ", dcc.Input(id='carrier_dev', value=0, type='number'),
              " Transmission SNR: ", dcc.Input(id='snr', value=30, type='number'),
              " Transmission Phase Deviation: ", dcc.Input(id='phase_dev', value=0, type='number'),
              html.Button(id='submit_state', n_clicks=0, children='Update'),],
             style={'textAlign':'center'}),
    html.Br(),

    html.Div(className='row', children=[
        dcc.Graph(figure={}, id='transmission')
    ]),
    html.Br(),

    html.Div(["Low Pass Frequency: ", dcc.Input(id='filter_freq', value=2_000, type='number'),
              " Filter Order: ", dcc.Input(id='filter_order', value=4, type='number'),
              html.Button(id='submit_state_filter', n_clicks=0, children='Update'),],
             style={'textAlign':'center'}),
    html.Br(),

    html.Div(className='row', children=[
        html.Div(className='six columns',children=[dcc.Graph(figure={}, id='message1_demod')]),
        html.Div(className='six columns',children=[dcc.Graph(figure={}, id='message2_demod')]),
    ]),

    html.Div(className='row', children=[
        html.Div(className='six columns',children=[dcc.Graph(figure={}, id='message1_demod_fft')]),
        html.Div(className='six columns',children=[dcc.Graph(figure={}, id='message2_demod_fft')]),
    ]),

    html.Div([html.Button(id='submit_state_sound_m1', n_clicks=0, children='Play M1'),
              html.Button(id='submit_state_sound_m2', n_clicks=0, children='Play M2'),],
             style={'textAlign':'center'}),
    html.Br(),

    dcc.Store(id='message1_data'),
    dcc.Store(id='message2_data'),
    dcc.Store(id='qam_data'),
    dcc.Store(id='message1_demod_data'),
    dcc.Store(id='message2_demod_data'),
    dcc.Store(id='trash'),
    dcc.Store(id='trash2'),
    dcc.Store(id='trash3'),
])

@callback(
    Output(component_id='message1_fig', component_property='figure'),
    Output(component_id='message1_data', component_property='data'),
    Input(component_id='submit_state_m1',component_property='n_clicks'),
    State(component_id='m1_f', component_property='value'),
)
def update_message1(submit_state,m1_f):
    t = np.arange(0, PERIOD, 1/FSH)
    df_message = pd.DataFrame(columns=["message","time"])
    if type(m1_f) is str:
        f = np.array(m1_f.split(',')).astype(float)
    else:
        f = m1_f
    df_message["message"] = message(f,t)
    df_message["time"] = t
    fig_m1 = go.Figure()
    fig_m1.add_trace(go.Scatter(x=df_message["time"][::int(FSH/FSL)], y=df_message["message"][::int(FSH/FSL)], name='Message 1 Signal'))
    fig_m1.update_layout(title='Message 1 Signal', xaxis_title='Time (s)', yaxis_title='Amplitude')
    return fig_m1,df_message.to_json(date_format="iso",orient="split")

@callback(
    Output(component_id='message2_fig', component_property='figure'),
    Output(component_id='message2_data', component_property='data'),
    Input(component_id='submit_state_m2',component_property='n_clicks'),
    State(component_id='m2_f', component_property='value'),
)
def update_message2(submit_state,m2_f):
    t = np.arange(0, PERIOD, 1/FSH)
    df_message = pd.DataFrame(columns=["message","time"])
    if type(m2_f) is str:
        f = np.array(m2_f.split(',')).astype(float)
    else:
        f = m2_f
    df_message["message"] = message(f,t)
    df_message["time"] = t
    fig_m2 = go.Figure()
    fig_m2.add_trace(go.Scatter(x=df_message["time"][::int(FSH/FSL)], y=df_message["message"][::int(FSH/FSL)], name='Message 2 Signal'))
    fig_m2.update_layout(title='Message 2 Signal', xaxis_title='Time (s)', yaxis_title='Amplitude')
    return fig_m2,df_message.to_json(date_format="iso",orient="split")

@callback(
    Output(component_id='transmission', component_property='figure'),
    Output(component_id='qam_data', component_property='data'),
    Input(component_id='submit_state',component_property='n_clicks'),
    Input(component_id='message1_data', component_property='data'),
    Input(component_id='message2_data', component_property='data'),
    State(component_id='m1_factor', component_property='value'),
    State(component_id='m2_factor', component_property='value'),
    State(component_id='carrier_freq', component_property='value'),
    State(component_id='carrier_dev', component_property='value'),
    State(component_id='snr', component_property='value'),
    State(component_id='phase_dev', component_property='value'),
)
def update_transmission(submit_state,
                        m1_message,m2_message,
                        m1_factor,m2_factor,
                        carrier_f,carrier_dev,
                        snr,phase_dev):

    message1 = pd.read_json(m1_message,orient="split")
    message2 = pd.read_json(m2_message,orient="split")
    df_qam = pd.DataFrame(columns=["message","time"])
    df_qam["time"] = message1["time"] if len(message1["time"]) > len(message2["time"]) else message2["time"]
    df_qam["message"] = QAM_modulation(message1["message"],carrier_f,df_qam["time"],message2=message2["message"],
                                       a1=m1_factor,a2=m2_factor,
                                       carrier_dev=carrier_dev, transmission_dev=phase_dev)

    noise_std = np.sqrt(np.sum(df_qam["message"]**2) / (len(df_qam["message"]) * 10**(snr/10)))  # calculate noise standard deviation
    noise = noise_std * (2*np.random.rand(len(df_qam["message"])) - 1)  # generate white noise
    df_qam["message_noisy"] = df_qam["message"] + noise  # add noise to QAM signal

    fft_y = fft(df_qam['message_noisy'].to_numpy())
    fft_x = fftfreq(len(df_qam['message_noisy'].to_numpy()),1/FSH)
    fig_m_fft = go.Figure()
    fig_m_fft.add_trace(go.Scatter(x=fft_x, y=np.abs(fft_y), name='Modulated Signal FFT'))
    fig_m_fft.update_layout(title='Modulated Signal FFT', xaxis_title='Frequency (Hz)', yaxis_title='Amplitude')
    fig_m_fft.update_xaxes(type="log")

    return fig_m_fft,df_qam.to_json(date_format="iso",orient="split")

@callback(
    Output(component_id='message1_demod_data', component_property='data'),
    Output(component_id='message2_demod_data', component_property='data'),

    Input(component_id='submit_state_filter',component_property='n_clicks'),
    Input(component_id='qam_data',component_property='data'),

    State(component_id='m1_factor', component_property='value'),
    State(component_id='m2_factor', component_property='value'),
    State(component_id='carrier_freq', component_property='value'),
    State(component_id='filter_freq', component_property='value'),
    State(component_id='filter_order', component_property='value'),
)
def update_demodulation(submit_state,qam_data,
                        m1_factor,m2_factor,
                        carrier_f,
                        filter_freq,filter_order):

    modulated_message = pd.read_json(qam_data, orient='split').astype('float32')
    demodulated_m1 = modulated_message.copy(deep=True).drop(columns=['message_noisy'])
    demodulated_m2 = modulated_message.copy(deep=True).drop(columns=['message_noisy'])
    demodulated_m1['message'],demodulated_m2['message'] = QAM_demodulation(carrier_f,modulated_message['time'],
                                m1_factor,m2_factor,modulated_message['message_noisy'],
                                fcutoff=filter_freq,order=filter_order)

    return demodulated_m1.to_json(date_format="iso",orient="split"),demodulated_m2.to_json(date_format="iso",orient="split")

@callback(
    Output(component_id='trash', component_property='data'),
    Input(component_id='message1_demod_data', component_property='data'),
    Input(component_id='message2_demod_data', component_property='data'),
)
def demodulated_sound(message1_data,message2_data):
    demodulated_m1 = pd.read_json(message1_data, orient='split').astype('float32')
    demodulated_m2 = pd.read_json(message2_data, orient='split').astype('float32')
    subsample_demod1 = demodulated_m1['message'][::int(FSH/FSL)]
    subsample_demod1 = np.int16(subsample_demod1 / np.max(np.abs(subsample_demod1) * np.iinfo(np.int16).max))
    subsample_demod2 = demodulated_m2['message'][::int(FSH/FSL)]
    subsample_demod2 = np.int16(subsample_demod2 / np.max(np.abs(subsample_demod2) * np.iinfo(np.int16).max))

    with wave.open('demod1.wav', 'w') as wave_file:
        wave_file.setnchannels(1)  # Mono audio
        wave_file.setsampwidth(2)  # 16-bit audio
        wave_file.setframerate(FSL)  # Samples per second
        wave_file.writeframes(subsample_demod1.astype(np.float32).tobytes())
    
    with wave.open('demod2.wav', 'w') as wave_file:
        wave_file.setnchannels(1)  # Mono audio
        wave_file.setsampwidth(2)  # 16-bit audio
        wave_file.setframerate(FSL)  # Samples per second
        wave_file.writeframes(subsample_demod2.astype(np.float32).tobytes())
    return None

@callback(
    Output(component_id='message1_demod', component_property='figure'),
    Output(component_id='message1_demod_fft', component_property='figure'),
    Input(component_id='message1_demod_data', component_property='data'),
)
def update_m1_demodulation(message1_data):
    df_message = pd.read_json(message1_data, orient='split')
    fig_m = go.Figure()
    fig_m.add_trace(go.Scatter(x=df_message['time'][::int(FSH/FSL)], y=df_message['message'][::int(FSH/FSL)], name='Message 1 Demodulated with Local Oscilator'))
    fig_m.update_layout(title='Message 1 Demodulated with Local Oscilator', xaxis_title='Time (s)', yaxis_title='Amplitude')

    fft_y = fft(df_message['message'].to_numpy())
    fft_x = fftfreq(len(df_message['message'].to_numpy()),1/FSH)
    fig_m_fft = go.Figure()
    fig_m_fft.add_trace(go.Scatter(x=fft_x, y=np.abs(fft_y), name='Message 1 Demodulated FFT'))
    fig_m_fft.update_layout(title='Message 1 Demodulated FFT', xaxis_title='Frequency (Hz)', yaxis_title='Amplitude')
    return fig_m,fig_m_fft

@callback(
    Output(component_id='message2_demod', component_property='figure'),
    Output(component_id='message2_demod_fft', component_property='figure'),
    Input(component_id='message2_demod_data', component_property='data'),
)
def update_m2_demodulation(message2_data):
    df_message = pd.read_json(message2_data, orient='split')
    fig_m = go.Figure()
    fig_m.add_trace(go.Scatter(x=df_message['time'][::int(FSH/FSL)], y=df_message['message'][::int(FSH/FSL)], name='Message 2 Demodulated with Local Oscilator'))
    fig_m.update_layout(title='Message 2 Demodulated with Local Oscilator', xaxis_title='Time (s)', yaxis_title='Amplitude')

    fft_y = fft(df_message['message'].to_numpy())
    fft_x = fftfreq(len(df_message['message'].to_numpy()),1/FSH)
    fig_m_fft = go.Figure()
    fig_m_fft.add_trace(go.Scatter(x=fft_x, y=np.abs(fft_y), name='Message 2 Demodulated FFT'))
    fig_m_fft.update_layout(title='Message 2 Demodulated FFT', xaxis_title='Frequency (Hz)', yaxis_title='Amplitude')
    return fig_m,fig_m_fft

@callback(
    Output(component_id='trash2', component_property='data'),
    Input(component_id='submit_state_sound_m1',component_property='n_clicks'),
)
def play_message1(submit_state):
    print("Playing message 1")
    # song = AudioSegment.from_wav("demod1.wav")
    # play(song)
    return "M1 Played"

@callback(
    Output(component_id='trash3', component_property='data'),
    Input(component_id='submit_state_sound_m2',component_property='n_clicks'),
)
def play_message2(submit_state):
    print("Playing message 2")
    # song = AudioSegment.from_wav("demod2.wav")
    # play(song)
    return "M2 Played"

def message(f,t):
    if hasattr(f,"__len__"):
        message = np.sum([np.sin(2*np.pi*fi*t) for fi in f],axis=0)
        print(message)
    else:
        message = np.sin(2*np.pi*f*t)
    return message

def QAM_modulation(message1,Fc,t,message2 = [],a1=0.5,a2=0.5,carrier_dev=0, transmission_dev=0):
    if hasattr(message2,"__len__") and len(message2)>0:
        if len(message1)>len(message2):
            message2.extend(np.zeros(len(message1)-len(message2)))
        elif len(message2)>len(message1):
            message1.extend(np.zeros(len(message2)-len(message1)))
    else:
        message2 = np.zeros_like(message1)

    Icarrier = np.sin(2*np.pi*Fc*t + transmission_dev*np.pi/2) # Carrier in phase
    Qcarrier = np.sin(2*np.pi*Fc*t + (1 + carrier_dev + transmission_dev)*np.pi/2) # Carrier in quadrature

    qam = (1 + a1*message1) * Icarrier + (1 + a2*message2) * Qcarrier
    return qam

def QAM_demodulation(Fc,t,a1,a2,signal,fcutoff=2_000,order=4):
    # Demodulate the QAM signal using a local oscillator
    local_oscilator = np.sin(2*np.pi*Fc*t) # Carrier in phase
    local_oscilator_q = np.sin(2*np.pi*Fc*t + np.pi/2) # Carrier in quadrature

    b, a = scipy.signal.butter(20, 2*np.pi*(Fc+20), fs=FSH, btype='high', analog=False)  # filter coefficients
    signal = scipy.signal.lfilter(b, a, signal)  # apply band-pass filter

    demod1 = a1 * signal * local_oscilator  # demodulate message1 signal
    demod2 = a2 * signal * local_oscilator_q  # demodulate message2 signal
    

    # Apply low-pass filters to remove high-frequency components
    wcutoff = 2*np.pi*fcutoff  # cutoff frequency for low-pass filter (radians/s)
    b, a = scipy.signal.butter(order, wcutoff, fs=FSH, btype='low', analog=False)  # filter coefficients

    demod1 = scipy.signal.lfilter(b, a, demod1)  # apply low-pass filter to demodulated message1 signal
    demod2 = scipy.signal.lfilter(b, a, demod2)  # apply low-pass filter to demodulated message2 signal

    return demod1.astype('float32'),demod2.astype('float32')

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)