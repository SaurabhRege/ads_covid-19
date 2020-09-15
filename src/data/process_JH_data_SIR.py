import pandas as pd
import numpy as np

from datetime import datetime


def store_large_flat_JH_data():
    ''' Transforms the raw COVID data into a large flat table structure
    
    '''
    
    datapath='C:/Users/Asus/ads_covid-19/data/raw/COVID-19/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv'
    pd_raw=pd.read_csv(datapath)
    
    time_index=pd_raw.columns[4:]
    pd_flat_table=pd.DataFrame({'date':time_index})
    
    country_list=pd_raw['Country/Region'].unique()
    
    for country in country_list:
        pd_flat_table[country]=np.array(pd_raw[pd_raw['Country/Region']==country].iloc[:,4::].sum(axis=0))
        
        
    time_idx=[datetime.strptime( each, "%m/%d/%y") for each in pd_flat_table.date]
    time_str=[each.strftime('%Y-%m-%d') for each in time_idx]
    
    pd_flat_table['date']=time_index
    
    pd_flat_table.to_csv('C:/Users/Asus/ads_covid-19/data/processed/COVID_large_flat_table.csv',sep=';',index=False )
    print('Latest date is'+str(max(pd_flat_table.date)))
    print(' Number of rows stored: '+str(pd_flat_table.shape[0]))

if __name__ == '__main__':
    store_large_flat_JH_data()

