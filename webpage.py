from altair.vegalite.v4.schema.channels import Opacity
from numpy.core.fromnumeric import size
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from streamlit_folium import folium_static
import folium as fl
from folium.plugins import MarkerCluster 
from folium.plugins import HeatMapWithTime
from folium.plugins import HeatMap
from urllib.request import urlopen
import warnings
import json
from plotly.subplots import make_subplots
import plotly.graph_objs as go
from multiapp import MultiApp
warnings.filterwarnings('ignore')
st.set_page_config(layout='wide')



############################################## DATASET DEFINITION ###############################################################################################################
@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def load_data():

    with open('Neighborhood Tabulation Areas (NTA).geojson') as response:
        districts1 = json.load(response)

    with urlopen('https://raw.githubusercontent.com/dwillis/nyc-maps/master/boroughs.geojson') as response:
        boroughs_NY = json.load(response)
    
    df_collis_copy=pd.read_pickle('./FINAL_DF_V11.pickle.xz', compression="xz")

    forecast_map_data=pd.read_pickle('forecast_map_data.pickle')

    forecast_plot=pd.read_pickle('forecast_plot.pickle')

    rain_data=pd.read_pickle('rain_data.pickle')

    df_collis_copy['BOROUGH'] = df_collis_copy['BOROUGH'].str.capitalize()
    df_collis_copy['BOROUGH'] = df_collis_copy['BOROUGH'].str.replace('Staten island','Staten Island')

    df_collis_copy['sum']= df_collis_copy['NUMBER OF PERSONS INJURED'] + df_collis_copy['NUMBER OF PERSONS KILLED']
    df_collis_copy['Collision type']= df_collis_copy['sum'].apply(lambda x: 'Collisions without injuries'if x==0 else None)

    df_collis_copy.loc[(df_collis_copy['NUMBER OF PERSONS INJURED']!=0) &(df_collis_copy['NUMBER OF PERSONS KILLED']==0),'Collision type']= 'Collisions with injuries'
    df_collis_copy.loc[(df_collis_copy['NUMBER OF PERSONS INJURED']==0) &(df_collis_copy['NUMBER OF PERSONS KILLED']!=0),'Collision type']= 'Lethal Collisions'

    df_collis_copy['VEHICLE TYPE CODE 1'] = df_collis_copy['VEHICLE TYPE CODE 1'].str.lower()
    df_collis_copy['CONTRIBUTING FACTOR VEHICLE 1'] = df_collis_copy['CONTRIBUTING FACTOR VEHICLE 1'].str.lower()

    df_collis_copy['30min']=df_collis_copy.datetime.dt.floor('30T').dt.time.astype(str)
    df_collis_copy['15min']=df_collis_copy.datetime.dt.floor('15T').dt.time.astype(str)
    df_collis_copy['day_hour']=df_collis_copy['DayofWeek']+' '+df_collis_copy.Hour.astype(str)
    df_collis_copy['Volume']=1

    def normalization_b(a_original):
        a = a_original.copy()
        b=df_collis_copy.pivot_table(index='BOROUGH',values='borough_pop',aggfunc='mean').T
        #top10=cnorm.neighborhood.to_list()
        for i in a.columns:
            try:
                if b[i].values[0]!=0:
                    a[i]=a[i]/b[i].values[0]
                else:
                    a[i]=0
            except:a[i]=0
        at=a.T
        at=at.reset_index()
        #at=at[(~at.BOROUGH.str.contains('park',case=False))]
        at['row_sum']=at.sum(axis=1)
        at=at[at.row_sum>0]
        at=at.drop(columns='row_sum')
        at=at.set_index('BOROUGH')
        at=at.T
        return at
    
    def normalization(a_original):
        a = a_original.copy()
        b=df_collis_copy.pivot_table(index='neighborhood',values='neigh_pop',aggfunc='mean').T
        for i in a.columns:
            try:
                if b[i].values[0]!=0:
                    a[i]=a[i]/b[i].values[0]
                else:
                    a[i]=0
            except:a[i]=0
        at=a.T
        at=at.reset_index()
        at=at[(~at.neighborhood.str.contains('park',case=False))]
        at['row_sum']=at.sum(axis=1)
        at=at[at.row_sum>0]
        at=at.drop(columns='row_sum')
        at=at.set_index('neighborhood')
        at=at.T
        return at

    ind=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday','Sunday']
    m_ind = ['Jan', 'Feb','Mar','Apr', 'May','Jun', 'Jul','Aug','Sep', 'Oct','Noe', 'Dec']

    
    top10 = ['Midtown-Midtown South',
    'Springfield Gardens South-Brookville',
    'Hudson Yards-Chelsea-Flatiron-Union Square',
    'Hunters Point-Sunnyside-West Maspeth',
    'Hunts Point',
    'Clinton',
    'Queensbridge-Ravenswood-Long Island City',
    'Turtle Bay-East Midtown',
    'East Williamsburg',
    'Rosedale']


    #dataframes for graphs
    df_1 = df_collis_copy['VEHICLE TYPE CODE 1'].value_counts().head(10) 
    df_2 = df_collis_copy['CONTRIBUTING FACTOR VEHICLE 1'].value_counts().head(10)

    cnorm=df_collis_copy.pivot_table(index='BOROUGH',values=['COLLISION_ID','borough_pop','NUMBER OF PERSONS INJURED','NUMBER OF PERSONS KILLED'],aggfunc=['count','mean','sum']).reset_index()
    cnorm.columns=[' '.join(col).strip() for col in cnorm.columns.values]
    cnorm=cnorm.rename(columns={'sum NUMBER OF PERSONS INJURED':'Total injured','mean NUMBER OF PERSONS INJURED':'Mean injured',
                           'mean NUMBER OF PERSONS KILLED':'Mean Killed',  
                           'sum NUMBER OF PERSONS KILLED':'Total killed',
                           'mean borough_pop':'Population',
                           'count COLLISION_ID':'Total Collisions'}).drop(['count NUMBER OF PERSONS KILLED','count borough_pop','count NUMBER OF PERSONS INJURED','sum borough_pop'],axis=1)

    b=df_collis_copy.pivot_table(index='BOROUGH',values='borough_pop',aggfunc='mean')
    df_3 =df_collis_copy.groupby('BOROUGH').agg('count')['Volume'].astype(float)
    for i in df_3.index:
        df_3[i]=df_3[i]/b['borough_pop'][i]
    
    df_4 = df_collis_copy.groupby(['year','BOROUGH']).agg('count')['Volume'].unstack()

    df_5 = normalization_b(df_collis_copy.groupby(['DayofWeek','Hour','BOROUGH']).count()['Volume'].unstack().reindex(ind,level=0)).reset_index()
    df_5['i'] = df_5['DayofWeek'] + ' ' +df_5['Hour'].astype(str)
    df_5 = df_5.set_index('i').drop(['DayofWeek','Hour'],axis=1)

    df_6 = normalization_b(df_collis_copy.groupby(['Hour','BOROUGH']).agg('count')['Volume'].unstack())


    cnorm2=df_collis_copy[df_collis_copy.neigh_pop>0].pivot_table(index=['neighborhood','BOROUGH','year','NUMBER OF PERSONS INJURED','NUMBER OF PERSONS KILLED'],values='neigh_pop',aggfunc=['mean','count']).reset_index()
    cnorm2.columns=[' '.join(col).strip() for col in cnorm2.columns.values]
    cnorm2=cnorm2.rename(columns={'mean neigh_pop':'neigh_pop','count neigh_pop':'collisions'})
    cnorm2['cn_KPI']=cnorm2.collisions/cnorm2.neigh_pop
    kpi_max=cnorm2.cn_KPI.max()
    kpi_min=cnorm2.cn_KPI.min()
    cnorm2['cn_KPI_nomalized']=cnorm2.cn_KPI.apply(lambda x: (x-kpi_min)/(kpi_max-kpi_min))
    temp=cnorm2.pivot_table(index='neighborhood',values=['NUMBER OF PERSONS INJURED','NUMBER OF PERSONS KILLED','cn_KPI','collisions','neigh_pop'],aggfunc=['sum','mean']).reset_index()#.sort_values(by='cn_KPI')
    temp.columns=[' '.join(col).strip() for col in temp.columns.values]
    temp=temp.rename(columns={'sum NUMBER OF PERSONS INJURED':'Total injured',
                            'sum NUMBER OF PERSONS KILLED':'Total killed',
                            'sum cn_KPI':'cn_KPI',
                            'sum collisions':'Total Collisions',
                            'mean NUMBER OF PERSONS INJURED':'Mean injured',
                            'mean NUMBER OF PERSONS KILLED':'Mean Killed',  
                            'mean collisions':'Mean Collisions',
                            'mean neigh_pop':'Population'})
    temp = temp.sort_values(by='cn_KPI')
    temp=temp[~temp.neighborhood.str.contains('park')]

    temp['Mean injured'] = temp['Mean injured'].astype(int)
    temp['Mean Collisions'] = temp['Mean Collisions'].astype(int)
    temp = temp[~temp.neighborhood.str.contains('park')]

    bb=df_collis_copy.pivot_table(index='neighborhood',values='neigh_pop',aggfunc='mean')
    df_7 =df_collis_copy[df_collis_copy.neighborhood.isin(top10)].groupby('neighborhood').agg('count')['Volume'].astype(float)
    for i in df_7.index:
        df_7[i]= df_7[i]/bb['neigh_pop'][i]
    
    df_8 = normalization(df_collis_copy[df_collis_copy.neighborhood.isin(top10)].groupby(['year','neighborhood']).agg('count')['Volume'].unstack())

    df_9 = normalization(df_collis_copy[df_collis_copy.neighborhood.isin(top10)].groupby(['DayofWeek','Hour','neighborhood']).count()['Volume'].unstack().reindex(ind,level=0))
    df_9= df_9.reset_index()
    df_9['i'] =df_9['DayofWeek'] + ' ' +df_9['Hour'].astype(str)

    df_10 = df_collis_copy.groupby(['Hour','neighborhood']).agg('count')['Volume'].unstack()
    df_10 = normalization(df_10)[top10]

    df_collis_copy['injured'] = df_collis_copy['NUMBER OF PERSONS INJURED'] + df_collis_copy['NUMBER OF PEDESTRIANS INJURED'] + df_collis_copy['NUMBER OF CYCLIST INJURED'] + df_collis_copy['NUMBER OF MOTORIST INJURED'] 
    df_collis_copy['killed']  = df_collis_copy['NUMBER OF PERSONS KILLED'] + df_collis_copy['NUMBER OF PEDESTRIANS KILLED'] + df_collis_copy['NUMBER OF CYCLIST KILLED'] + df_collis_copy['NUMBER OF MOTORIST KILLED'] 

    df_11 = df_collis_copy.groupby('year').agg('sum')[['injured','killed']].reset_index()
    
    df_12 = df_collis_copy.groupby('month').agg('sum')[['injured','killed']].reindex(m_ind).reset_index()

    df_13 = df_collis_copy.groupby('Hour').agg('sum')[['injured','killed']].reset_index()

    df_14 = df_collis_copy.groupby(['DayofWeek','Hour']).agg('sum')[['injured','killed']].reindex(ind,level=0).reset_index()
    df_14['i'] =df_14['DayofWeek'] + ' ' +df_14['Hour'].astype(str)

    return df_collis_copy, df_1, df_2, cnorm, boroughs_NY, districts1, df_3, df_4, df_5, df_6, temp, df_7, df_8, df_9, top10, df_10, df_11, df_12, df_13, df_14, forecast_map_data, forecast_plot, rain_data

############################################ VIZUAL DEFINITION #################################################################################################################

ind=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday','Sunday']
m_ind = ['Jan', 'Feb','Mar','Apr', 'May','Jun', 'Jul','Aug','Sep', 'Oct','Noe', 'Dec']
height_in=450
width_in=900


df, df_1, df_2, cnorm, boroughs_NY, districts1,df_3, df_4, df_5, df_6, temp, df_7, df_8, df_9, top10, df_10, df_11, df_12, df_13, df_14, forecast_map_data, forecast_plot, rain_data= load_data() 

@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def normalization_b(a_original):
    a = a_original.copy()
    b=df.pivot_table(index='BOROUGH',values='borough_pop',aggfunc='mean').T
    #top10=cnorm.neighborhood.to_list()
    for i in a.columns:
        try:
            if b[i].values[0]!=0:
                a[i]=a[i]/b[i].values[0]
            else:
                a[i]=0
        except:a[i]=0
    at=a.T
    at=at.reset_index()
    #at=at[(~at.BOROUGH.str.contains('park',case=False))]
    at['row_sum']=at.sum(axis=1)
    at=at[at.row_sum>0]
    at=at.drop(columns='row_sum')
    at=at.set_index('BOROUGH')
    at=at.T
    return at

#dt       
@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def normalization(a_original):
    a = a_original.copy()
    b=df.pivot_table(index='neighborhood',values='neigh_pop',aggfunc='mean').T
    for i in a.columns:
        try:
            if b[i].values[0]!=0:
                a[i]=a[i]/b[i].values[0]
            else:
                a[i]=0
        except:a[i]=0
    at=a.T
    at=at.reset_index()
    at=at[(~at.neighborhood.str.contains('park',case=False))]
    at['row_sum']=at.sum(axis=1)
    at=at[at.row_sum>0]
    at=at.drop(columns='row_sum')
    at=at.set_index('neighborhood')
    at=at.T
    return at


@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def hm_time(df,timestep):
    plotdata = []
    temp=df
    temp.loc[:,'count']=1
    for hour in temp[timestep].sort_values().unique():
        plotdata.append(temp.loc[temp[timestep] == hour, ['LONGITUDE', 'LATITUDE', 'count']].groupby(['LATITUDE','LONGITUDE']).sum().reset_index().values.tolist())

    time_index= temp[timestep].sort_values().unique().tolist()
    base_map = fl.Map([40.700610, -73.905242],zoom_start=11,tiles = "Stamen Toner")

    HeatMapWithTime(plotdata, index=time_index, auto_play=True, radius=4,
                    gradient={0.1: 'blue', 0.4: 'lime',
                            0.8: 'orange', 1: 'red'},
                    min_opacity=0.5, max_opacity=0.8, 
                    use_local_extrema=True, min_speed=2.5).add_to(base_map)
    return base_map

period=hm_time(df,timestep='period')
min=hm_time(df,timestep='15min')




#-------------------------------------------------------------- 1st page graphs --------------------------------------------------------------------------------------

width_in_1st = 950

fig1 = px.pie(df['Collision type'].value_counts().reset_index() ,values='Collision type', names='index',width=1700, height=height_in,title='Collision types')


fig2 = px.bar(df[['NUMBER OF PEDESTRIANS INJURED','NUMBER OF CYCLIST INJURED','NUMBER OF MOTORIST INJURED']].sum(),
             color=['NUMBER OF PEDESTRIANS INJURED','NUMBER OF CYCLIST INJURED','NUMBER OF MOTORIST INJURED'],width=width_in_1st, height=450,title='Injured')
fig2.update_layout(yaxis_title="Collisions",legend_title="Category", xaxis_title="", legend=dict(
    orientation="h",
    yanchor="bottom",
    y=1.02,
    xanchor="right",
    x=1
))


fig3 = px.bar(df[['NUMBER OF PEDESTRIANS KILLED','NUMBER OF CYCLIST KILLED','NUMBER OF MOTORIST KILLED']].sum(),
             color=['NUMBER OF PEDESTRIANS KILLED','NUMBER OF CYCLIST KILLED','NUMBER OF MOTORIST KILLED'],width=width_in_1st, height=450,title='Killed')
fig3.update_layout(yaxis_title="Collisions",legend_title="Category", xaxis_title="", legend=dict(
    orientation="h",
    yanchor="bottom",
    y=1.02,
    xanchor="right",
    x=1
))

fig4 = px.bar(df_1,color=df_1.index,width=width_in_1st, height=600,title='Top 10 Vehicle Types')
fig4.update_layout(yaxis_title="Collisions",legend_title="Category",legend_bgcolor= 'rgba(0,0,0,0)', xaxis_title="",yaxis_range=[0.0,500000], legend=dict(
    orientation="h", 
    yanchor="bottom",
    y=0.8,
    xanchor="right",
    x=1.001
))

fig5 = px.bar(df_2,
             color=df_2.index,
             width=width_in_1st, 
             height=600,
             title='Top 10 Contributing Factors')
fig5.update_layout(yaxis_title="Collisions", legend_title="Category", legend_bgcolor= 'rgba(0,0,0,0)', xaxis_title="", legend=dict(
    orientation="h", 
    yanchor="bottom",
    y=0.75,
    xanchor="right",
    x=1.001
))

fig16 = make_subplots(specs=[[{"secondary_y": True}]])
fig16.add_trace(go.Line(x=df_11['year'], y=df_11['injured'], name="number of injured"),secondary_y=False,)
fig16.add_trace(go.Line(x=df_11['year'], y=df_11['killed'], name="number of killed"),secondary_y=True,)
fig16.update_layout(title_text='Yearly injured-killed by collisions', width=width_in_1st, height=600, legend_bgcolor= 'rgba(0,0,0,0)',legend=dict(
    orientation="h",
    yanchor="bottom",
    y=1.02,
    xanchor="right",
    x=1
))
fig16.update_yaxes(title_text="number of injured", secondary_y=False)
fig16.update_yaxes(title_text="number of killed", secondary_y=True)


fig17 = make_subplots(specs=[[{"secondary_y": True}]])
fig17.add_trace(go.Line(x=df_12['month'], y=df_12['injured'], name="number of injured"),secondary_y=False,)
fig17.add_trace(go.Line(x=df_12['month'], y=df_12['killed'], name="number of killed"),secondary_y=True,)
fig17.update_layout(title_text='The monthly patterns for injured-killed by collisions', width=width_in_1st, height=600, legend_bgcolor= 'rgba(0,0,0,0)',legend=dict(
    orientation="h",
    yanchor="bottom",
    y=1.02,
    xanchor="right",
    x=1
))
fig17.update_yaxes(title_text="number of injured", secondary_y=False)
fig17.update_yaxes(title_text="number of killed", secondary_y=True)


fig18 = make_subplots(specs=[[{"secondary_y": True}]])
fig18.add_trace(go.Line(x=df_13['Hour'], y=df_13['injured'], name="number of injured"),secondary_y=False,)
fig18.add_trace(go.Line(x=df_13['Hour'], y=df_13['killed'], name="number of killed"),secondary_y=True,)
fig18.update_layout(xaxis_title="Hour",title_text='The 24-hour cycle for injured-killed by collisions', width=width_in_1st, height=600, legend_bgcolor= 'rgba(0,0,0,0)',legend=dict(
    orientation="h",
    yanchor="bottom",
    y=1.02,
    xanchor="right",
    x=1
))
fig18.update_yaxes(title_text="number of injured", secondary_y=False)
fig18.update_yaxes(title_text="number of killed", secondary_y=True)


fig19 = make_subplots(specs=[[{"secondary_y": True}]])
fig19.add_trace(go.Line(x=df_14['i'], y=df_14['injured'], name="number of injured"),secondary_y=False,)
fig19.add_trace(go.Line(x=df_14['i'], y=df_14['killed'], name="number of killed"),secondary_y=True,)
fig19.update_layout(title_text='Hours of the week for injured-killed by collisions', width=width_in_1st, height=600, legend_bgcolor= 'rgba(0,0,0,0)',legend=dict(
    orientation="h",
    yanchor="bottom",
    y=1.02,
    xanchor="right",
    x=1
))
fig19.update_yaxes(title_text="number of injured", secondary_y=False)
fig19.update_yaxes(title_text="number of killed", secondary_y=True)

#-------------------------------------------------------------- 2nd page graphs --------------------------------------------------------------------------------------

fig6 = px.choropleth_mapbox(cnorm, geojson=boroughs_NY,locations='BOROUGH',
                           color='Total Collisions',
                           featureidkey="properties.BoroName",
                           color_continuous_scale="Viridis",
                           hover_data=['BOROUGH', 'Total Collisions', 'Mean injured', 'Mean Killed',
                                       'Population', 'Total injured', 'Total killed'],
                           #range_color=(0, 2000),
                           mapbox_style="carto-positron",
                           zoom=9, center = {"lat": 40.730610, "lon": -73.935242},
                           opacity=.5, width=1700)
fig6.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

fig7 = px.bar(df_3,x=df_3.index,y='Volume',color=df_3.index,width=width_in, height=height_in,title='Collisions per Borough (Normalized)')
fig7.update_layout(xaxis_title="",yaxis_title="Collisions per population",legend_bgcolor= 'rgba(0,0,0,0)', legend=dict(
    orientation="h", 
    yanchor="bottom",
    y=1.02,
    xanchor="right",
    x=1
))

fig8 = px.line(normalization_b(df_4),width=width_in, height=height_in,title='Yearly Trend per Borough (Normalized)')
fig8.update_layout(yaxis_title="Collisions per population", legend_bgcolor= 'rgba(0,0,0,0)', legend=dict(
    orientation="h", 
    yanchor="bottom",
    y=1.02,
    xanchor="right",
    x=1
))

fig9 = px.line(df_5,width=width_in, height=height_in,title='The 24-7 cycle per Borough (Normalized)')
fig9.update_layout(xaxis_title="",yaxis_title="Collisions per population", legend_bgcolor= 'rgba(0,0,0,0)', legend=dict(
    orientation="h", 
    yanchor="bottom",
    y=1.02,
    xanchor="right",
    x=1
))

fig10 = px.line(df_6,width=width_in, height=height_in,title='The 24-hour cycle per Borough (Normalized)')
fig10.update_layout(xaxis_title="Hour",yaxis_title="Collisions per population", legend_bgcolor= 'rgba(0,0,0,0)', legend=dict(
    orientation="h", 
    yanchor="bottom",
    y=1.02,
    xanchor="right",
    x=1
))

#-------------------------------------------------------------- 3rd page graphs --------------------------------------------------------------------------------------

fig11 = px.choropleth_mapbox(temp,geojson=districts1,locations='neighborhood',color='cn_KPI',
                    featureidkey="properties.ntaname",
                    color_continuous_scale="Viridis",
                     hover_name='neighborhood',
                     hover_data=['cn_KPI', 'Total injured', 'Total killed', 'Total Collisions', 'Mean injured', 'Mean Killed','Mean Collisions', 'Population'],
                    range_color=(0, 0.4),
                    mapbox_style="carto-positron",
                    zoom=9.5, center = {"lat": 40.730610, "lon": -73.935242},
                    opacity=.5,width=1700, height=700)


fig12 = px.bar(df_7,x=df_7.index,y='Volume',width=width_in, height=600,title='Collisions per neighborhood (Normalized)')
fig12.update_layout(xaxis_title="",yaxis_title="Collisions per population")

fig13 = px.line(df_8,width=width_in, height=600,title='Yearly Trend per neighborhood')
fig13.update_layout(yaxis_title="Collisions per population", legend_bgcolor= 'rgba(0,0,0,0)',yaxis_range=[0.0,0.24], legend=dict(
    orientation="h", 
    yanchor="bottom",
    y=0.70,
    xanchor="right",
    x=1.001
))

fig14 = px.line(df_9,x='i',y=top10,width=width_in, height=600,title='The 24-7 cycle per neighborhood (Normalized)')
fig14.update_layout(xaxis_title="",yaxis_title="Collisions per population",legend_bgcolor= 'rgba(0,0,0,0)',yaxis_range=[0.0,0.02], legend=dict(
    orientation="h", 
    yanchor="bottom",
    y=0.72,
    xanchor="right",
    x=1.001
))

fig15 = px.line(df_10,width=width_in, height=600,title='The 24-hour cycle per neighborhood (Normalized)')
fig15.update_layout(xaxis_title="Hour",yaxis_title="Collisions per population",legend_bgcolor= 'rgba(0,0,0,0)',yaxis_range=[0.0,0.12], legend=dict(
    orientation="h", 
    yanchor="bottom",
    y=0.74,
    xanchor="right",
    x=1.001
))

#-------------------------------------------------------------- 4th page graphs --------------------------------------------------------------------------------------

fig20 = px.choropleth_mapbox(forecast_map_data,geojson=districts1,locations='neighborhood',color='labels_2',
                        featureidkey="properties.ntaname",
                        labels={"labels_2": "Collision Forecast"},
                        category_orders={"labels_2": ["Increase", "Decrease", "Stable"]},
                        color_discrete_map={ "Increase": "green", "Decrease": "red",'Stable':'blue'},
                        mapbox_style="carto-positron",
                        zoom=9.5, center = {"lat": 40.730610, "lon": -73.935242},
                        opacity=.5,title='Collisions Forecast for 2021 R-Square: 0.7',width=1700, height=700)



fig21 = px.line(forecast_plot,x="date", y=["Actual","If Corona didn't happen"],
              title="Collision Forecast if Corona didn't happened - R-square: 0.94 (before Corona)", width=1700, height=500)

fig22 = px.line(rain_data,x='i',y=['No Rain','Rain','Heavy Rain'],width=1700, height=500,title='The 24-7 cycle for the neighborhood with an increase in collisions influenced by weather conditions')
fig22.update_layout(xaxis_title="",yaxis_title="Collisions-weather influence")

############################################## WEBPAGE DESIGN ##################################################################################################################



def app1():
    st.title("Social Data Analysis and Vizualization - New York Collisions Analysis")
    st.write("Welcome :) This dashboard was created in the framework of Social Data Analysis and Vizualization course for the Technical University of Denmark."+
            " In this dashboard, we are presenting trafic collisions that have happened in New York city. On the left side of the page, there is a navigation pane that can be used to go through each section of the dashboard.")
    st.write("**DISCLAIMER** It is recommended to use white theme while you go through the dashboard. You can always adjust your theme on the top right area of the page using settings.")
    
    col1,col2 = st.beta_columns([1.12,1])
    col1.image('https://images.squarespace-cdn.com/content/v1/564be6bde4b0884e9478a03f/1578496563576-9NEBMM8OGF1M36U8KU6L/ke17ZwdGBToddI8pDm48kNvT88LknE-K9M4pGNO0Iqd7gQa3H78H3Y0txjaiv_0fDoOvxcdMmMKkDsyUqMSsMWxHk725yiiHCCLfrh8O1z5QPOohDIaIeljMHgDF5CVlOqpeNLcJ80NK65_fV7S1UU_i9-ln4sAC0TGEmkfMFKJn5Kcyb6Y0O9dBHu3N61jtpC969RuPXvt2ZwyzUXQf7Q/NY-Night-Cinemagraph.gif?format=2500w', width=933)
    col2.image('https://wallpapercave.com/wp/WuKIyI5.jpg', width=700)
    #col1.video('https://youtu.be/tYQ1Okyi3g4') 
    st.markdown("""The main goal of this report is to analyse the collisions that happened in New York City the last 7 years and indetify patterns and insights. 

The datasets that were used are:

* [New York Motor Vehicle Collisions](https://data.cityofnewyork.us/Public-Safety/Motor-Vehicle-Collisions-Crashes/h9gi-nx95)
* [New York Neighborhood Areas](https://data.cityofnewyork.us/City-Government/Neighborhood-Tabulation-Areas-NTA-/cpf4-rkhq)
* [New York Population per Neighborhood](https://dev.socrata.com/foundry/data.cityofnewyork.us/swpk-hqdp)
* [Weather Data](https://www.noaa.gov/)

The design behind the UI presented in the webpage is based upon a multi-page comic book genre style. Each page is devided into the respective sections of the analysis which are:

* Welcome Page 
* Collision Analysis 
* Borough Analysis
* Neighborhood Analysis
* Collision Forecasting
* Conclusions and Discussions

The motivation behind the project is to understand the seasonality of the collisions that are happening in NYC and investigate if there are any neighborhoods with a high number of collisions relatively to their population.
 Afterwards, if such neighborhoods are indetified,  a more proactive plan of placing the emergency services around the city can be made.

GitHub [link](https://github.com/IcyDingo/Social_Data_2021?fbclid=IwAR2Z1CPXIY-el7RIKhyz99HhcrmpFURWT8ejiT57u20bMZObIupBuC363q0) with the final datasets and supproting Python Files.

Google Drive [link](https://drive.google.com/file/d/1dIj6iRJ_zx4BPfL86HrP4v3Bd9pEG38x/view?usp=sharing) to the notebook.""",unsafe_allow_html=True)


    
#@st.cache(suppress_st_warning=True)
def app2():
    
    st.title("Collision Analysis")
    st.write("In this part of the dashboard, a high level analysis is executed on the accidents that have happened in New York city for the period of 2014 until 2020."
            +" The first graph shows how many accidents had injuries, deaths and also how many of those collisions were without an injury."
            +" Given the fact that New York is one of the biggest cities in the world, having injuries in 20% of the accidents means that this is a serious problem for NYC."
            +" In this project, we want to raise awairness for this issue and find ways to reduce both percentage of injuries in accidents and total number of accidents.")

    st.plotly_chart(fig1)
    st.write("To begin with, the following 6 graphs describe the accidents that had injuries or deaths for the studied period. ")
    st.markdown("""Highlighted Resuls:

- According to the first two digrams, the most injuries happen to motorists, meaning people that have a motorvehicle, which makes sense, because, most of the times, they are the ones
 involved in a collision. However, the most deaths occur to pedestrian. This might be the case because 
motorists, being inside the car, are more protected compared to pedestrians and cyclists that are way more exposed than the former.
- According to yearly pattern, it can be seen that the number of injuries increased between 2016 and 2019 compared to previous years. 
- Between 2014 and 2018 an downward trend can be seen for the number of persons who died except of the year 2017. In 2017, a terrorist attack happended in NYC resulting in 18 deaths. [Source](https://www.vox.com/2017/10/31/16587120/new-york-city-terror-attack-what-we-know) 
- The number of deaths is increasing over the weekend and over the nights, which can be explained of the alcohol consumption during night-life. 
- In 2020, an overall decline can be observed, due to the COVID-19 pandemic.
- According to the monthly patterns, the first 3 months of the year (January-March) have the least number for both injuries and deaths.

Finally, the last two graphs describe the cars that are involved in **all** traffic collisions and the contributing factors respectively:

- The most common vehicles involved in collisions are Sedans and passenger vehicles. 
- The most common contributing factors are the distraction/inattention of the driver, driving too close, and failure to yield right-of-the-way.""",unsafe_allow_html=True)
    col1,col2 = st.beta_columns([1,1])
    #with col1:
    #col1.header("**Hey1**")S
    col1.plotly_chart(fig2)
    col1.plotly_chart(fig16)
    col1.plotly_chart(fig18)
    #col1.title("Alex")
    col1.plotly_chart(fig4)
     

    #with col2:
    #col2.title("Hey2")
    col2.plotly_chart(fig3)
    col2.plotly_chart(fig17)
    col2.plotly_chart(fig19)
    #col2.title('YES')
    col2.plotly_chart(fig5)
    
#@st.cache(allow_output_mutation=True)
def app3():
    st.title('Borough Analysis')
    st.markdown("""In this part of the dashboard the 5 main boroughs of New York (Manhattan, Broklyn, Bronx, Staten Islandm, Quenns) are analyzed. The purpose of this page is to 
    identify the most crusial borough of New York according to the seasonality analysis. 
In the maps below, it can be observed that the borough with the most collisions during 2014-2020 is the _Brooklyn_ with a total of almost 430,000 collision, 520 people dead and 
about 130,000 people were injured. For the Seasonality Analysis, it is important to normalize the number of collisions per borough relatively to their population.
 Normalization plays an important role, because the distributions change significantly among the boroughs. However, only the normalized results are displayed here. 
 For more information visit the notebook.""")
    st.plotly_chart(fig6)
    st.write("")
    st.write("The map below diplays the heatmap of the collisions in the whole New York city on a monthly basis throughout the years 2014-2020. It can be observed that Manhattan and Brooklyn has the highest density of collisions, wheareas, Staten Island has the smallest density.")
    folium_static(period,width=1700, height=600)
    st.markdown("""Highlighted results:
- **Manhattan** is the borough with the most collisions  per population and **Brooklyn** is the borough with the most collisions in total.
- Over the years, on each there is a slight increase of the  total collisions. However, a downward trend can be abserved after 2019, which can be explained from the COVID-19 pandemic outbursting. 
- According to 24-7 cycle and the 24-hours cycle, the number of collisions drop over the weekend and during the night. In contrast, an upward tendecy can be observed during the 
afternoons and the the week days. This can be explained because on week days most of citizens are going to their jobs and during the afternoon they return from there. 
- According to the monthly seasonality, it can be seen that during the spring there is a slight decrease on the collisions compared to the winter season. April and February are the months with the least amount of collisions on each Borough respectively. """)
    col1,col2 = st.beta_columns([1,1])
    with col1:
        #col1.header("**Hey1**")
    #st.text(day_week)
    #with st.beta_expander("See explaination"):
    #    st.write('This explains the dataset')
    #    st.write("This is also explaining the dataset")
        #col1.plotly_chart(fig1)
        #col1.title("Alex")
        col1.plotly_chart(fig7)
        col1.plotly_chart(fig9)
        

    with col2:
        #col2.title('YES')
        col2.plotly_chart(fig8)
        col2.plotly_chart(fig10)
        

def app4():
    st.title('Neighborhood Analysis')
    st.markdown("""In this section, 193 neighboorhoods of New York city were analyzed. From the Borough Analysis, it was pointed out that Manhattan is the most crusial borough. However, 
    the Boroughs cover a huge amount of area and thus, it was decided to go one level down and analyze the neighborhoods individually. 

All neighborhoods, that are displayed in the map below, have been colored scaled and normallized according to the number of collisisons per population. The color scale corresponds to 
the KPI factor, the higher the factor the greater the collisions per population. The most important ones are the yellow areas which are: 

- Midtown-Midtown South
- Springfield Gardens South-Brookville
- Hudson Yards-Chelsea-Flatiron-Union Square
- Hunters Point-Sunnyside-West Maspeth
- Hunts Point
- Clinton
- Queensbridge-Ravenswood-Long Island City
- Turtle Bay-East Midtown
- East Williamsburg
- Rosedale

These areas are the most crucial and among them during 2014-2020 is the Midtown Manhattan with a total of 332,290 collisions, 11 people died, and 181 were injured.""")
    st.plotly_chart(fig11)
    st.write("The heatmap below, displays the the 24-hour cycle. It can be seen, that the downtown is area of interest and it peaks during the afternoon, mostly because people are returning home from their workplace.")
    folium_static(min, width=1700, height=700)
    st.markdown("""For the Seasonality Analysis,the collisions has been normalized with the same procedure as the Borough Analysis and only the nomralized results are displayed. 
    For more information on how the normalization is achieved and on how the non-normalized results were generated, check the notebook. To avoid clutter, 
    only the 10 top neighborhoods according to the KPI factors are visualized in the seasonality analysis.

**Highlighted Results:**

- **Midtown** stands out from the other neighborhoods. This can be explained, because of the low number of residents and high number of offices. 
- Between 2014 and 2018, a slight increase of the total collisions per neighboorhood can be observed. However, a downward trend can be seen after 2019, which can be explained from the COVID-19 pandemic. 
- According to 24-7 cycle and the 24-hours cycle, the collisions drop over weekends and during the night. In contrast, an upward tendecy can be observed during the afternoons and the the week days. This can be explained because, on the week days most of citizens are going to their worklplace and  during the afternoon they return from there. 
- According to the monthly seasonality, it can be seen that during  spring there is a slight decrease on the collisions compared to the winter season. April and February are the months with the least amount of collisions on each neighborhood, respectively. """)

    col1,col2 = st.beta_columns([1,1])

    with col1:
        st.plotly_chart(fig12)
        st.plotly_chart(fig14)

    with col2:
        st.plotly_chart(fig13)
        st.plotly_chart(fig15)

def app5():
    st.title("Collision Forecasting")
    st.markdown("""The main idea behind forecasting collisions is to proactively plan the placing of emergency services around the city (police patrols and ambulances)

The time series analysis was done using [FB Prophet](https://facebook.github.io/prophet/) which a high level API based on [Stan](https://mc-stan.org/) which uses Statistical Modeling techniques for time-series forecasting. [Paper](https://peerj.com/preprints/3190/)

The forecasting was done in 2 main phases:
* Forecasting for the whole NYC before corona
* Forecasting for each NYC Neighborhood

For the first graph, The reasoning behind forecasting collisions for the whole city, before corona, is to get a holistic view of what would have happened regarding traffic collisions 
if there was no corona pandemic. It is clear from the graph that if the pandemic did not happend the collisions would follow on a slightly increasing trend. That's **123.000** less collisions during year 2020.""", unsafe_allow_html=True)

    st.plotly_chart(fig21)
    st.markdown("""The forecasting per neighborhood was done in order to identify which neighborhoods have an increasing, decreasing or stable trend. 
    The comparison was made to 2018 for the actual values wich is consider to be a normal year. The neighborhood with an increasing trend can be categorised as a 
    possible hot spot area were more collisions are forecasted to happen. The areas which are forecasted to have an increasing amount of trafic collisions during the year 2021 
    are the **Mid-Town Manhattan** and the south part of **Saten Island**. Acording to this results more emphasis should be put on those areas regading emergency services placing in 
    the city in order to aid proactivness. 

It is important to note that that the **Saten Island** area was not a hot spot on on the previous descriptive analysis and based on the model it is expected to have an 
increased amount of collsions. A test of this forecasting approach would be to check the actual collision result once the year 2021 is concluded and compare them to the 
forecasted ones.""")
    st.plotly_chart(fig20)
    st.markdown("""Taking into consideration the 24-7 hour cycle of those neighborhood, weighted by the weather influence, we can provide an aproximation of where collsions are most likely to increase, what time and in which weather condition. 
    Looking at the graph bellow, on Wednesdays during mid-day there is a higher influence if it is raining heavyly. This information would indicate that, in the highlighted 
    neighborhoods from the map above, there is a higher collision probability in those areas. Based on this approach, it can now be indicated not only the location but also the 
    time and weather condition where the emergenices services could be placed proactively. 

Regarding the weather influence, it is assumed that when there is rain the chance of an accident is higher. Based on this assumption a weight was calculated for each collision. 
The higher the precipitation levels the higher weight.   """)
    st.plotly_chart(fig22)
    
def app6():
    st.title("Conclusions and Discussions")
    st.markdown("""The main takeaways from this project are:
* Collisions follow a highly predictable trend with high seasonality as shown from the analysis.
* There is a very small precentage of people that are killed in an accident in NYC (less that 0.1%).
* The hotspot collision area of the city is Midtown Manhattan. 
* Most collisions happen during mid-day hours but the most fatal ones happened at night, probably due to alchool consumption.
* During the week the activity is more or less the same, peaking in Fridays. In the weekends the activity drops about 25% but fatal accidents increase by 20%.
* Fatal accident include mostly pedestrians and not motorists.
* 1 out of 5 accidents include an injury.
* Collision prediction can be used in aidding emergency services proactiveness.
* Normalizing the collision according to population of the neighborhood reveals a clearer view of the situation since it's an apples to apples comparison ;). 

Predicting collision on the exact location is a difficult task but it can be made easier by making a prediction for each neighborhod. The model that was used has an r-square of 0.95
 which a very good performance. This is possible due the exploitation of seasonality in the dataset and the use of FB Prophet, which is working better when there are strong 
 seasonality patterns. Since it's a time-series problem extra fetures like weather and vehicle type can not be included in this model. Another model that takes this features 
 into consideration could have been a choice, but, because of the seasonality factor, FB Prophet was chosen. The way that the weather data was treated on this analysis comes with 
 the assumption that, when it's raining the probability of a collision increases, which might not be the case or might have a smaller correlation. Further, a more thorough analysis on the 
 weather conection to collisions can be made in a future project.""")
    st.header("And let us not forget about these type of collisions...")
    st.image("https://i.pinimg.com/originals/81/da/f3/81daf3b1d9621a05b1910d6083d1f630.gif", width=933)

app = MultiApp()

app.add_app("Welcome Page", app1)
app.add_app("Collision Analysis",app2)
app.add_app("Borough Analysis",app3)
app.add_app("Neighborhood Analysis", app4)
app.add_app("Collision Forecasting", app5)
app.add_app("Conclusions and Discussions", app6)

app.run()