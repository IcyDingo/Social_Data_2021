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


############################################## DATASET DEFINITION ###############################################################################################################
@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def load_data():

    with open('Neighborhood Tabulation Areas (NTA).geojson') as response:
        districts1 = json.load(response)

    with urlopen('https://raw.githubusercontent.com/dwillis/nyc-maps/master/boroughs.geojson') as response:
        boroughs_NY = json.load(response)
    
    df_collis_copy=pd.read_pickle('./FINAL_DF_V9.pickle')

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


    return df_collis_copy, df_1, df_2, cnorm, boroughs_NY, districts1, df_3, df_4, df_5, df_6, temp, df_7, df_8, df_9, top10, df_10, df_11

############################################ VIZUAL DEFINITION #################################################################################################################

ind=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday','Sunday']
m_ind = ['Jan', 'Feb','Mar','Apr', 'May','Jun', 'Jul','Aug','Sep', 'Oct','Noe', 'Dec']
height_in=450
width_in=900


df, df_1, df_2, cnorm, boroughs_NY, districts1,df_3, df_4, df_5, df_6, temp, df_7, df_8, df_9, top10, df_10, df_11= load_data() 

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

fig1 = px.pie(df['Collision type'].value_counts().reset_index() ,values='Collision type', names='index',width=1700, height=height_in,title='Collision types')


fig2 = px.bar(df[['NUMBER OF PEDESTRIANS INJURED','NUMBER OF CYCLIST INJURED','NUMBER OF MOTORIST INJURED']].sum(),
             color=['NUMBER OF PEDESTRIANS INJURED','NUMBER OF CYCLIST INJURED','NUMBER OF MOTORIST INJURED'],width=width_in, height=450,title='Injured')
fig2.update_layout(yaxis_title="Collisions",legend_title="Category", legend=dict(
    orientation="h",
    yanchor="bottom",
    y=1.02,
    xanchor="right",
    x=1
))


fig3 = px.bar(df[['NUMBER OF PEDESTRIANS KILLED','NUMBER OF CYCLIST KILLED','NUMBER OF MOTORIST KILLED']].sum(),
             color=['NUMBER OF PEDESTRIANS KILLED','NUMBER OF CYCLIST KILLED','NUMBER OF MOTORIST KILLED'],width=width_in, height=450,title='Killed')
fig3.update_layout(yaxis_title="Collisions",legend_title="Category", legend=dict(
    orientation="h",
    yanchor="bottom",
    y=1.02,
    xanchor="right",
    x=1
))

fig4 = px.bar(df_1,color=df_1.index,width=width_in, height=600,title='Top 10 Vehicle Types')
fig4.update_layout(yaxis_title="Collisions",legend_title="Category",legend_bgcolor= 'rgba(0,0,0,0)',yaxis_range=[0.0,500000], legend=dict(
    orientation="h", 
    yanchor="bottom",
    y=0.8,
    xanchor="right",
    x=1.001
))

fig5 = px.bar(df_2,
             color=df_2.index,
             width=width_in, 
             height=600,
             title='Top 10 Contributing Factors')
fig5.update_layout(yaxis_title="Collisions", legend_title="Category", legend_bgcolor= 'rgba(0,0,0,0)', legend=dict(
    orientation="h", 
    yanchor="bottom",
    y=0.7,
    xanchor="right",
    x=1.001
))

fig16 = make_subplots(specs=[[{"secondary_y": True}]])
fig16.add_trace(go.Line(x=df_11['year'], y=df_11['injured'], name="number of injured"),secondary_y=False,)
fig16.add_trace(go.Line(x=df_11['year'], y=df_11['killed'], name="number of killed"),secondary_y=True,)
fig16.update_layout(title_text='Yearly injured-killed by collisions', width=width_in, height=600, legend_bgcolor= 'rgba(0,0,0,0)',legend=dict(
    orientation="h",
    yanchor="bottom",
    y=1.02,
    xanchor="right",
    x=1
))
fig16.update_yaxes(title_text="number of injured", secondary_y=False)
fig16.update_yaxes(title_text="number of killed", secondary_y=True)

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
fig7.update_layout(xaxis_title="",yaxis_title="Collisions",legend_bgcolor= 'rgba(0,0,0,0)', legend=dict(
    orientation="h", 
    yanchor="bottom",
    y=1.02,
    xanchor="right",
    x=1
))

fig8 = px.line(normalization_b(df_4),width=width_in, height=height_in,title='Yearly Trend per Borough (Normalized)')
fig8.update_layout(yaxis_title="Count of collisions", legend_bgcolor= 'rgba(0,0,0,0)', legend=dict(
    orientation="h", 
    yanchor="bottom",
    y=1.02,
    xanchor="right",
    x=1
))

fig9 = px.line(df_5,width=width_in, height=height_in,title='The 24-7 cycle per Borough (Normalized)')
fig9.update_layout(xaxis_title="",yaxis_title="Collisions", legend_bgcolor= 'rgba(0,0,0,0)', legend=dict(
    orientation="h", 
    yanchor="bottom",
    y=1.02,
    xanchor="right",
    x=1
))

fig10 = px.line(df_6,width=width_in, height=height_in,title='The 24-hour cycle per Borough (Normalized)')
fig10.update_layout(xaxis_title="Hour",yaxis_title="Collisions", legend_bgcolor= 'rgba(0,0,0,0)', legend=dict(
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
fig12.update_layout(xaxis_title="",yaxis_title="Collisions")

fig13 = px.line(df_8,width=width_in, height=600,title='Yearly Trend per neighborhood')
fig13.update_layout(yaxis_title="Count of collisions", legend_bgcolor= 'rgba(0,0,0,0)',yaxis_range=[0.0,0.24], legend=dict(
    orientation="h", 
    yanchor="bottom",
    y=0.70,
    xanchor="right",
    x=1.001
))

fig14 = px.line(df_9,x='i',y=top10,width=width_in, height=600,title='The 24-7 cycle per neighborhood (Normalized)')
fig14.update_layout(xaxis_title="",yaxis_title="Collisions",legend_bgcolor= 'rgba(0,0,0,0)',yaxis_range=[0.0,0.02], legend=dict(
    orientation="h", 
    yanchor="bottom",
    y=0.72,
    xanchor="right",
    x=1.001
))

fig15 = px.line(df_10,width=width_in, height=600,title='The 24-hour cycle per neighborhood (Normalized)')
fig15.update_layout(xaxis_title="Hour",yaxis_title="Collisions",legend_bgcolor= 'rgba(0,0,0,0)',yaxis_range=[0.0,0.12], legend=dict(
    orientation="h", 
    yanchor="bottom",
    y=0.74,
    xanchor="right",
    x=1.001
))

############################################## WEBPAGE DESIGN ##################################################################################################################


def app1():
    st.title("Social Data Analysis and Vizualization")
    st.write("Welcome :) This dashboard was created in the framework of Social Data Analysis and Vizualization course for the Technical University of Denmark."+
            " In this dashboard, we are presenting trafic collisions that have happened in New York city. On the left side of the page there is a navigation pane that can be used to go through each section of the dashboard. ")
    col1,col2 = st.beta_columns([1.12,1])
    col1.image('https://images.squarespace-cdn.com/content/v1/564be6bde4b0884e9478a03f/1578496563576-9NEBMM8OGF1M36U8KU6L/ke17ZwdGBToddI8pDm48kNvT88LknE-K9M4pGNO0Iqd7gQa3H78H3Y0txjaiv_0fDoOvxcdMmMKkDsyUqMSsMWxHk725yiiHCCLfrh8O1z5QPOohDIaIeljMHgDF5CVlOqpeNLcJ80NK65_fV7S1UU_i9-ln4sAC0TGEmkfMFKJn5Kcyb6Y0O9dBHu3N61jtpC969RuPXvt2ZwyzUXQf7Q/NY-Night-Cinemagraph.gif?format=2500w', width=933)
    col2.image('https://wallpapercave.com/wp/WuKIyI5.jpg', width=700)
    #col1.video('https://youtu.be/tYQ1Okyi3g4') 
    
#@st.cache(suppress_st_warning=True)
def app2():
    
    st.title("Descriptive Analysis")
    st.write("In this part of the dashboard, we are doing a high level analysis on the accidents that have happened in New York city for the last 7 years."
            +" The first piechart shows how many accidents contained injuries and deaths and also how many of those collisions were without an injury."
            +" Given the fact that New York is one of the biggest cities in the world, having injuries in 20% of the accidents means that this is a serious problem for New York city."
            +" In this project, we want to raise awairness for this issue and find ways to reduce both the percentage of injuries in accidents and also the total number of accidents.")

    st.plotly_chart(fig1)
    col1,col2 = st.beta_columns([1,1])
    #with col1:
    #col1.header("**Hey1**")S
    col1.plotly_chart(fig2)
    col1.plotly_chart(fig16)
    #col1.title("Alex")
    col1.plotly_chart(fig4)
     

    #with col2:
    #col2.title("Hey2")
    col2.plotly_chart(fig3)
    #col2.title('YES')
    col2.plotly_chart(fig5)
    
#@st.cache(allow_output_mutation=True)
def app3():
    st.title('Borough Analysis')
    st.plotly_chart(fig6)
    folium_static(period,width=1700, height=600)
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
    st.plotly_chart(fig11)
    folium_static(min, width=1700, height=700)
    col1,col2 = st.beta_columns([1,1])

    with col1:
        st.plotly_chart(fig12)
        st.plotly_chart(fig14)

    with col2:
        st.plotly_chart(fig13)
        st.plotly_chart(fig15)
    


app = MultiApp()

app.add_app("Welcome Page", app1)
app.add_app("Descriptive Analysis",app2)
app.add_app("Borough Analysis",app3)
app.add_app("Neighborhood Analysis", app4)

app.run()