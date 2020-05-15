# Load, clean, and combine shape files of hunger scores in East Africa.
# Combine this with subnational boundaries for aggregate hunger scores for each
# region-season.

# Matt Mauer
# 2020-05-14 

import pandas as pd
import geopandas as gpd
import numpy as np

# read in the previously wrangled districts dataset
regions = gpd.read_file('./regions.geojson')

# create year and month indicies for reading files and generating time features
years = [x+2000 for x in range(9,20)]
months = range(1,13)

scores = gpd.GeoDataFrame()
cnt = 0

# Read in all score files and append them to empty scores gdf
for year in years:
    for month in months:
        
        # format the file request by time period of recorded hunger scores
        if month < 10:
            period = str(year) + '0' + str(month)
        else:
            period = str(year) + str(month)
        
        # read the file in if it exists
        try:
            period_scores = gpd.read_file('./FEWSNET/East Africa/EA_' + \
                                          period + '_CS.shp')
        except:
            continue
            
        # add time columns
        period_scores['year'] = year
        period_scores['month'] = month
        # append to the total dataframe
        scores = scores.append(period_scores, sort=True)
        cnt += 1

# Join the scores and region to get scores at the district level
region_scores = gpd.sjoin(regions, scores, how='left', op='intersects')

# drp unnecessary data
region_scores = region_scores[region_scores.CS <= 5]
region_scores = region_scores.drop(['HA0', 'index_right'],axis=1)

# gat the average scores for each region-season
region_scores_agg = region_scores.groupby(by=['ADMIN0', 'ADMIN1', \
                                                'year', 'month'], \
                                                sort=False).mean()

# join aggregate df back to gdf for spatial data
region_scores_final = region_scores.merge(region_scores_agg, \
                                          on=['ADMIN0', 'ADMIN1', \
                                              'year', 'month'])

# clean
region_scores_final['CS'] = region_scores_final['CS_y']
region_scores_final = region_scores_final.drop(['CS_x', 'CS_y'], axis=1)
region_scores_cleaned = region_scores_final.drop_duplicates()

# write
region_scores_final.to_file('region_scores.geojson', driver='GeoJSON')

print(f'{cnt} CS files combined with our admin boundaries file into one \
    aggregate score file')
