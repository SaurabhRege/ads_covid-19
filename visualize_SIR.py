import pandas as pd
import numpy as np
import plotly.graph_objects as go
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input,Output

import os
print(os.getcwd())
df_input_large=pd.read_csv('C:/Users/Asus/ads_covid-19/data/processed/COVID_large_flat_table.csv',sep=';',parse_dates=[0])
df_input_large=df_input_large.sort_values('date',ascending=True)
df_input_SIR=pd.read_csv('C:/Users/Asus/ads_covid-19/data/processed/COVID_large_fitted_table.csv',sep=';')
df_input_SIR=df_input_SIR.sort_values('date',ascending=True)

fig=go.Figure()
app=dash.Dash()

app.layout=html.Div([
        dcc.Markdown('''
    #  Applied Data Science on COVID-19 data

    Goal of the project is to teach data science by applying a cross industry standard process,
    it covers the full walkthrough of: automated data gathering, data transformations,
    filtering and machine learning to approximating the doubling time, and
    (static) deployment of responsive dashboard.

    '''),

    dcc.Markdown('''
    ## Select a Country for Visualization of a Simulated SIR Curve
    '''),

        
        dcc.Dropdown( id='country_drop_down_simulated_list',
                     options=[{'label':each,'value':each} for each in df_input_large.columns[1:]],
                     value='Germany',
                     multi=False),
    
    #Manipulating the values of beta ,gamma, t_initial, t_intro_measures,t_hold,t_relax to achieve the simulated curve

    dcc.Markdown('''
        ## Vary the different values to reshape the SIR curve(Enter a number and press Enter)
        '''),
    
    html.Label(["Infection rate in days, when no measure introduced",             
    dcc.Input(
    id='t_initial',
    type='number',
    value=28,debounce=True)]),

    html.Br(),
    html.Br(),

    html.Label(["Infection rate in days, when measure introduced",             
    dcc.Input(
    id='t_intro_measures',
    type='number',
    value=14,debounce=True)]),

    html.Br(),
    html.Br(),

    html.Label(["Infection rate in days, when measure sustained/held",             
    dcc.Input(
    id='t_hold',
    type='number',
    value=21,debounce=True)]),

    html.Br(),
    html.Br(),

    html.Label(["Infection rate in days, when measure relaxed/removed",             
    dcc.Input(
    id='t_relax',
    type='number',
    value=21,debounce=True)]),
    

    html.Br(),
    html.Br(),
    
    html.Label(["Beta max",             
    dcc.Input(
    id='beta_max',
    type='number',
    value=0.4,debounce=True)]),

    html.Br(),
    html.Br(),
    
    html.Label(["Beta min",
    dcc.Input(
    id='beta_min',
    type='number',
    value=0.1,debounce=True)]),

    html.Br(),
    html.Br(),
    
    html.Label(["Gamma",             
    dcc.Input(
    id='gamma',
    type='number',
    value=0.1,debounce=True)]),

    html.Br(),
    html.Br(),
    
    dcc.Graph(figure=fig, id='SIR_simulated', animate=False,),
    
    dcc.Markdown('''
    ## Select a Country for Visualization of a Fitted SIR Curve
    '''),
    
    dcc.Dropdown( id='country_drop_down_fitted_list',
                     options=[{'label':each,'value':each} for each in df_input_SIR.columns[1:]],
                     value='Germany',
                     multi=False),
    
    dcc.Graph(id='SIR_fitted', animate=False,)
    
        ])
        
    
@app.callback(
    Output('SIR_simulated', 'figure'),
    [Input('country_drop_down_simulated_list', 'value'),
    Input('t_initial','value'),
    Input('t_intro_measures','value'),
    Input('t_hold','value'),
    Input('t_relax','value'),
    Input('beta_max','value'),
    Input('beta_min','value'),
    Input('gamma','value')])
    
def update_figure(country,t_initial, t_intro_measures, t_hold, t_relax, beta_max, beta_min, gamma):
    ydata=df_input_large[country][df_input_large[country]>=30]
    xdata=np.arange(len(ydata))
    N0=10000000
    I0=30
    S0=N0-I0
    R0=0
    gamma    
    SIR=np.array([S0,I0,R0])
    
    t_initial
    t_intro_measures
    t_hold
    t_relax
    beta_max
    beta_min
    propagation_rates=pd.DataFrame(columns={'susceptible':S0,'infected':I0,'recovered':R0})
    pd_beta=np.concatenate((np.array(t_initial*[beta_max]),
                       np.linspace(beta_max,beta_min,t_intro_measures),
                       np.array(t_hold*[beta_min]),
                       np.linspace(beta_min,beta_max,t_relax),
                       ))
    
    def SIR_model(SIR,beta,gamma):
        'SIR model for simulatin spread'
        'S: Susceptible population'
        'I: Infected popuation'
        'R: Recovered population'
        'S+I+R=N (remains constant)'
        'dS+dI+dR=0 model has to satisfy this condition at all time'
        S,I,R=SIR
        dS_dt=-beta*S*I/N0
        dI_dt=beta*S*I/N0-gamma*I
        dR_dt=gamma*I
        return ([dS_dt,dI_dt,dR_dt])
    
    for each_beta in pd_beta:
        new_delta_vec=SIR_model(SIR,each_beta,gamma)
        SIR=SIR+new_delta_vec
        propagation_rates=propagation_rates.append({'susceptible':SIR[0],'infected':SIR[1],'recovered':SIR[2]},ignore_index=True) 
    
    fig=go.Figure()
    fig.add_trace(go.Bar(x=xdata,
                        y=ydata,
                         marker_color='red',
                         name='Confirmed Cases'                
                        ))
    
    fig.add_trace(go.Scatter(x=xdata,
                            y=propagation_rates.infected,
                            mode='lines',
                            marker_color='blue',
                            name='Simulated curve'))
    
    fig.update_layout(shapes=[
                            dict(type='rect',xref='x',yref='paper',x0=0,y0=0,x1=t_initial,y1=1,fillcolor="midnightblue",opacity=0.3,layer="below"),
                            dict(type='rect',xref='x',yref='paper',x0=t_initial,y0=0,x1=t_initial+t_intro_measures,y1=1,fillcolor="midnightblue",opacity=0.4,layer="below"),
                            dict(type='rect',xref='x',yref='paper',x0=t_initial+t_intro_measures,y0=0,x1=t_initial+t_intro_measures+t_hold,y1=1,fillcolor="midnightblue",opacity=0.5,layer='below'),
                            dict(type='rect',xref='x',yref='paper',x0=t_initial+t_intro_measures+t_hold,y0=0,x1=t_initial+t_intro_measures+t_hold+t_relax,y1=1,fillcolor="midnightblue",opacity=0.6,layer='below')
                            ],
                    title='SIR Simulation Scenario',
                    title_x=0.5,
                    xaxis=dict(title='Timeline',
                               titlefont_size=16),
                    yaxis=dict(title='Confirmed infected people (source johns hopkins csse, log-scale)',
                               type='log',
                                titlefont_size=16,
                              ),
                    width=1600,
                    height=900,
                     )
    return fig

@app.callback(
    Output('SIR_fitted', 'figure'),
    [Input('country_drop_down_fitted_list', 'value')])
    
    
def SIR_figure(country_list):
    df_SIR= df_input_SIR
    
    for n in df_SIR[1:]:
        data = []
        trace = go.Scatter(x=df_SIR.date,
                           y=df_SIR[country_list],
                           mode='lines+markers',
                           name = country_list)
        data.append(trace)
        
        trace_fit = go.Scatter(x=df_SIR.date,
                                  y=df_SIR[country_list +'_fitted'], 
                                  mode='lines+markers',
                                  name=country_list+'_fitted')
        data.append(trace_fit)
        
        
            
    return {'data': data,
            'layout' : dict(
                width=1600,
                height=900,
                title= 'SIR Fitted Curve',
                xaxis={'tickangle':-45,
                        'nticks':20,
                        'tickfont':dict(size=14,color="#7f7f7f"),
                      },
                yaxis={'type':"log"
                      }
                
            )
        } 

if __name__ == '__main__':
    app.run_server(debug=True,use_reloader=False)