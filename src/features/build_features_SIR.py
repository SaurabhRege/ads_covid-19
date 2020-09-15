import pandas as pd
import numpy as np
from scipy import optimize
from scipy import integrate

df_large_flat=pd.read_csv('C:/Users/Asus/ads_covid-19/data/processed/COVID_large_flat_table.csv',sep=';').iloc[80:]

df_list = df_large_flat.columns
df_list = list(df_list)

# Functions for SIR model
def SIR_model_t(SIR,t,beta,gamma):
    ''' Simple SIR model
        S: susceptible population
        t: time step, mandatory for integral.odeint
        I: infected people
        R: recovered people
        beta: 
        
        overall condition is that the sum of changes (differnces) sum up to 0
        dS+dI+dR=0
        S+I+R= N (constant size of population)
    
    '''
    
    S,I,R=SIR
    dS_dt=-beta*S*I/N0          #S*I is the 
    dI_dt=beta*S*I/N0-gamma*I
    dR_dt=gamma*I
    return dS_dt,dI_dt,dR_dt

#Function defined for optimize curve fit

def fit_odeint(x, beta, gamma):
    '''
    helper function for the integration
    '''
    return integrate.odeint(SIR_model_t, (S0, I0, R0), t, args=(beta, gamma))[:,1] # we only would like to get dI

#Fitting parameter for SIR model
for each in df_list[1:]:
    ydata = np.array(df_large_flat[each])
    t=np.arange(len(ydata))
    
    N0 = 10000000 #max susceptible population 
    I0=ydata[0]
    S0=N0-I0
    R0=0

    popt, pcov = optimize.curve_fit(fit_odeint, t, ydata, maxfev = 1000)
    perr = np.sqrt(np.diag(pcov))

    # get the final fitted curve
    fitted=fit_odeint(t, *popt).reshape(-1,1)
    df_large_flat[each +'_fitted'] = fitted 
    
df_large_flat.to_csv('C:/Users/Asus/ads_covid-19/data/processed/COVID_large_fitted_table.csv', sep=';',index=False)