from datetime import datetime

import folium
import geopandas
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from folium.plugins import MarkerCluster
from streamlit_folium import folium_static

# Page layout settings
st.set_page_config(layout='wide')


# Function to read data from the file
@st.cache_data
def get_data(path):
    data = pd.read_csv(path)
    return data


# Read the geopandas file defining the quadrants for density by price
@st.cache_data
def get_geofile(geo_data):
    geofile = geopandas.read_file(geo_data)

    return geofile


# Create a new column price_m2 (price per square meter)
def set_feature(data):
    data['price_m2'] = data['price'] / data['sqft_lot']
    return data


# Convert date column to a readable format
def set_date(data):
    data['date'] = pd.to_datetime(data['date'], format='%Y%m%dT%H%M%S').dt.strftime('%Y-%m-%d')
    return data


# Function to display a table with Plotly
def plotly_table(results, rows, style):
    header_values = ['<b>' + item.capitalize() + '</b>' for item in results.columns]
    cell_values = [results.iloc[:, index : index + 1] for index in range(len(results.columns))]

    if not style:
        st.dataframe(results.head())
    else:
        fig = go.Figure(data=[go.Table(
            header=dict(values=header_values,
                        fill_color='lightgrey',
                        align='center',
                        font=dict(size=12)),
            cells=dict(values=cell_values,
                       fill_color='whitesmoke',
                       align='center',
                       font=dict(size=12)))
        ])

        st.plotly_chart(fig, use_container_width=True)


# Selection of rows and display style for the table
def select_number_of_rows_and_if_style(results: pd.DataFrame):
    style = st.checkbox("Style dataframe?", False)
    rows = st.selectbox(
        "Select number of table rows to display",
        options=[5, 10, 50, 100, 500, 1000, 5000, 10000, 50000, len(results)],
    )
    return rows, style


def overview_data(data):
    # Sidebar for attribute and zip code selection
    st.sidebar.title('Data Overview')
    f_attributes = st.sidebar.multiselect('Select Attributes', data.columns)
    f_zipcode = st.sidebar.multiselect('Select Zip Code', data['zipcode'].unique())

    st.title('Data Overview')
    st.subheader("Explore and analyze the dataset based on chosen "
                 "_attributes_ and _zip codes_ or via _random sampling_ for broader insights.")

    # Organize visual and avoid #N/A
    if (f_zipcode != []) & (f_attributes != []):
        df_atr = data.loc[data['zipcode'].isin(f_zipcode), ['id', 'zipcode'] + f_attributes]

    elif (f_zipcode != []) & (f_attributes == []):
        df_atr = data.loc[data['zipcode'].isin(f_zipcode), :]

    elif (f_zipcode == []) & (f_attributes != []):
        df_atr = data.loc[:, ['id', 'zipcode'] + f_attributes]
    else:
        df_atr = data.copy()
    rows, style = select_number_of_rows_and_if_style(df_atr)
    if rows and style:
        plotly_table(df_atr.head(rows), rows, style)
    else:
        st.dataframe(df_atr.head(rows), hide_index=True)

    # Define two side-by-side columns
    c1, c2 = st.columns([1, 1])
    if (f_zipcode != []):
        filtered_data = data.loc[data['zipcode'].isin(f_zipcode)]


        # Average metrics by zipcode
        # - Total number of properties
        df1 = filtered_data[['id', 'zipcode']].groupby('zipcode').count().reset_index()
        # - Average price
        df2 = filtered_data[['price', 'zipcode']].groupby('zipcode').mean().reset_index()
        # - Average living room size
        df3 = filtered_data[['sqft_living', 'zipcode']].groupby('zipcode').mean().reset_index()
        # - Average price per square meter
        df4 = filtered_data[['price_m2', 'zipcode']].groupby('zipcode').mean().reset_index()


        # Merge
        m1 = pd.merge(df1, df2, on='zipcode', how='inner')
        m2 = pd.merge(m1, df3, on='zipcode', how='inner')
        df = pd.merge(m2, df4, on='zipcode', how='inner')

        df.columns = ['Zip Code', 'Number of properties', 'Price (avg)', 'Living room size (avg)', 'Price/m2 (avg)']

        c1.header('Average Values')
        c1.write("Displaying average values based on selected Zip Code:")
        c1.dataframe(df, height=600, width=800, hide_index=True)

        # Descriptive Statistics
        num_attributes = filtered_data.select_dtypes(include=['int64', 'float64']).drop(columns=['id'])
        mean = pd.DataFrame(num_attributes.apply(np.mean))
        median = pd.DataFrame(num_attributes.apply(np.median))
        std = pd.DataFrame(num_attributes.apply(np.std))
        max_attributes = pd.DataFrame(num_attributes.apply(np.max))
        min_attributes = pd.DataFrame(num_attributes.apply(np.min))

        df1 = pd.concat([max_attributes, min_attributes, mean, median, std], axis=1).reset_index()
        df1.columns = ['attributes', 'max', 'min', 'mean', 'median', 'std']

        c2.header('Descriptive Analysis')
        c2.write("Displaying descriptive statistics based on selected Zip Code:")
        c2.dataframe(df1, height=600, width=800, hide_index=True)
    else:
        filtered_data = data.sample(10, replace=True)
        c1.subheader("Values")
        c1.write("No Zip Code selected. Values of 10 random samples:")
        c1.dataframe(filtered_data.sample(10), height=600, width=800, hide_index=True)

        c2.subheader("Descriptive Analysis")
        c2.write("No Zip Code selected. Descriptive statistics for 10 random samples:")
        descriptive_stats = filtered_data.describe().T[['max', 'min', 'mean', '50%', 'std']]
        descriptive_stats.reset_index(inplace=True)
        descriptive_stats.columns = ['attributes', 'max', 'min', 'mean', 'median', 'std']
        c2.dataframe(descriptive_stats, height=600, width=800, hide_index=True)


def portfolio_density_maps(data, geofile):
    # Density of the portfolio - maps to see the amount of properties by region
    st.title('Region Overview')
    st.subheader("Explore geographic distribution of the portfolio. The data displayed is a _sample of 100 random points_, showing their distribution and price density across the region.")

    data_sample = data.sample(100)
    c1, c2 = st.columns(2, gap="large")
    c1.header('Portfolio Density')
    # Base Map: empty - Folium
    density_map = folium.Map(location=[data_sample['lat'].mean(),
                                       data_sample['long'].mean()],
                             default_zoom_start=15)

    # Adding points to the map
    marker_cluster = MarkerCluster().add_to(density_map)
    for name, row in data_sample.iterrows():
        folium.Marker([row['lat'], row['long']],
                      popup='Sold USD{0} on: {1}. Features: {2} sqft, {3} bedrooms, {4} bathrooms, year built: {5}'.format(
                          row['price'],
                          row['date'],
                          row['sqft_living'],
                          row['bedrooms'],
                          row['bathrooms'],
                          row['yr_built'])).add_to(marker_cluster)

    # Folium is not native to streamlit, so I need to put this with clause to make possible to plot the map
    with c1:
        folium_static(density_map, width=500)

    # Region Price Map
    c2.header('Price Density')
    # For each one of the zipcodes, an average price was calculated
    data_sample['zipcode'] = data_sample['zipcode'].astype(str)
    data_sample['zipcode'] = data_sample['zipcode'].str.replace(',', '')
    df = data_sample[['price', 'zipcode']].groupby('zipcode').mean().reset_index()
    df.columns = ['ZIP', 'PRICE']

    # Need to make a filter, to not have the whole geofile been plotted only regions that I have in df
    geofile = geofile[geofile['ZIPCODE'].isin(df['ZIP'].tolist())]

    region_price_map = folium.Map(location=[data_sample['lat'].mean(),
                                            data_sample['long'].mean()],
                                  default_zoom_start=15)

    # Make a density plot by color
    folium.Choropleth(data=df,
                      geo_data=geofile,
                      columns=['ZIP', 'PRICE'],
                      key_on='feature.properties.ZIPCODE',
                      fill_color='YlOrRd',
                      fill_opacity=0.7,
                      line_opacity=0.2,
                      legend_name='Average Price').add_to(region_price_map)

    with c2:
        folium_static(region_price_map, width=500)
    return None


def tendency_data(data):
    # Properties distribution by commercial categories
    st.sidebar.title('Tendency Attributes')
    st.title('Tendency Analysis')
    st.subheader("Explore trends in the dataset, such as average price per year built, per day, and the distribution "
                 "of prices.")

    # Average price per Year
    data['date'] = pd.to_datetime(data['date']).dt.strftime('%Y-%m-%d')
    # Filters
    min_year_built = int(data['yr_built'].min())
    max_year_built = int(data['yr_built'].max())

    f_year_built = st.sidebar.slider('Max Year Built', min_year_built, max_year_built,
                                     min_year_built)

    st.header('Average Price per Year Built')
    st.subheader("Select a Maximum Year Built")
    # Data selection
    df = data.loc[data['yr_built'] < f_year_built]
    df = df[['yr_built', 'price']].groupby('yr_built').mean().reset_index()

    # Plot
    fig = px.line(df, x='yr_built', y='price')
    st.plotly_chart(fig, use_container_width=True)

    # Average price per Day
    st.header('Average Price per Day')
    st.subheader("Select a Maximum Data")


    #  Filters
    min_date = datetime.strptime(data['date'].min(), '%Y-%m-%d')
    max_date = datetime.strptime(data['date'].max(), '%Y-%m-%d')

    f_date = st.sidebar.slider('Max Date', min_date, max_date, min_date)

    # Data filtering
    data['date'] = pd.to_datetime(data['date'])
    df = data.loc[data['date'] < f_date]
    df = df[['date', 'price']].groupby('date').mean().reset_index()

    # Plot
    fig = px.line(df, x='date', y='price')
    st.plotly_chart(fig, use_container_width=True)

    # Histogram
    st.header('Price Distribution')
    st.subheader("Select a Maximum Price")

    # filters
    min_price = int(data['price'].min())
    max_price = int(data['price'].max())
    avg_price = int(data['price'].mean())

    # data filtering
    f_price = st.sidebar.slider('Max Price', min_price, max_price, avg_price)
    df = data.loc[data['price'] < f_price]

    # data plot
    fig = px.histogram(df, x='price', nbins=50)
    st.plotly_chart(fig, use_container_width=True)

    return None


def attributes_distribution(data):
    # Properties distribution by categories
    st.sidebar.title('House Attributes')
    st.title('Property Attributes Distribution')
    st.subheader("Explore the distribution of houses based on _selected attribute criteria_ like bedrooms, bathrooms, floors, and water view.")

    # filters
    f_bedrooms = st.sidebar.selectbox('Max number of bedrooms',
                                      sorted(set(data['bedrooms'].astype(int).unique())),
                                  format_func=lambda x: int(x))
    f_bathrooms = st.sidebar.selectbox('Max number of bethrooms',
                                       sorted(set(data['bathrooms'].astype(int).unique())),
                                  format_func=lambda x: int(x))

    c1, c2 = st.columns([1, 1])

    # House per bedrooms
    c1.header('Houses per bedrooms')
    df = data[data['bedrooms'] < f_bedrooms]
    fig = px.histogram(df, x='bedrooms', nbins=19)
    c1.plotly_chart(fig, use_container_width=True)

    # House per bathrooms
    c2.header('Houses per bathrooms')
    df = data[data['bathrooms'] < f_bathrooms]
    fig = px.histogram(df, x='bathrooms', nbins=19)
    c2.plotly_chart(fig, use_container_width=True)

    # filters
    f_floors = st.sidebar.selectbox('Max number of floors',
                                    sorted(set(data['floors'].astype(int).unique())),
                                  format_func=lambda x: int(x))
    f_waterview = st.sidebar.checkbox('Only houses with water view?')

    c1, c2 = st.columns([1, 1])

    # House per floors
    c1.header('Houses per floor')
    df = data[data['floors'] < f_floors]

    fig = px.histogram(df, x='floors', nbins=19)
    c1.plotly_chart(fig, use_container_width=True)

    # House with waterview
    c2.header('Houses with water view?')

    fig = go.Figure()

    # Adiciona barras de quantidade de casas com e sem vista para o mar
    if f_waterview:
        df = data[data['waterfront'] == 1]
        count_with_view = len(df)
        fig.add_trace(go.Bar(x=['With Water View'], y=[count_with_view], marker_color='lightblue', text=[count_with_view], textposition='inside'))
        fig.update_layout(
            title='Quantity of Houses with Water View',
            yaxis_title='Quantity'
        )
        fig.update_traces(textfont=dict(size=24))
    else:
        df = data[data['waterfront'] == 0]
        count_without_view = len(df)
        fig.add_trace(go.Bar(x=['Without Water View'], y=[count_without_view], marker_color='lightgray', text=[count_without_view], textposition='inside'))
        fig.update_layout(
            title='Quantity of Houses without Water View',
            yaxis_title='Quantity'
        )
        fig.update_traces(textfont=dict(size=24))

    c2.plotly_chart(fig, use_container_width=True)

    return None


if __name__ == '__main__':
    # Data extraction
    data_path = 'kc_house_data.csv'
    geo_path = 'Zip_Codes.geojson'

    # Get data
    data = get_data(data_path)

    # Get geofile
    geofile = get_geofile(geo_path)

    # Data transformation
    data = set_feature(data)
    data = set_date(data)

    overview_data(data)

    portfolio_density_maps(data, geofile)

    tendency_data(data)

    attributes_distribution(data)

    # Data loading
