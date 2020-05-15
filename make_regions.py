# Pull local shape files of administrative zones in East Africa 
# obtained at https://fews.net/fews-data/334
# Write out single shape file of pertinent data

# Matt Mauer 
# 2020-05-14

import pandas as pd
import geopandas as gpd
import numpy as np

# list identifiers for each country folder
admin_files = ['ET_Admin1_2014', 'SD_Admin1_2014', 'SS_Admin1_2011', 'UG_Admin1_2014', \
               'KE_Admin1_2013', 'SO_Admin1_1990', 'TZ_Admin1_2012']
regions = gpd.GeoDataFrame()

# assemble dataframe
for country in admin_files:
        country_gdf = gpd.read_file('zip:///Users/matthewmauer/ML/project/admin_zones/' + \
                                 country + '.zip!' + country + '.shp')
        regions = regions.append(country_gdf, sort=True)

# select columns and write out
regions = regions[['geometry', 'ADMIN0', 'ADMIN1']]
regions.to_file('regions.geojson', driver='GeoJSON')