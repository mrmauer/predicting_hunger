# Predicting hunger and food insecurity

The order in which the notebooks were created and ideally should be run is as follows:
1. create_scores.ipynb (Creates hunger scores as column from FEWS shape data files)
2. make_regions.ipynb (Creates geodataframes for Horn of Africa sub-regions)
3. conflict/acled_wrangling.ipynb (Conflict data for regions in the HoA)
4. create_targets/create_targets.ipynb (Creates targets with 1,2 and 3 month offsets)
5. food_prices/wfp_wrangling.ipynb (Food price data cleaned into a dataframe)
6. population/population_wrangling.ipynb (demographic data from WB and govts)
7. weather/Weather Data.ipynb (Weather data from GHCN)
8. data_exploration_ipynb (data exploration and viz)
9. gather_data.ipynb (all datasets merged for modelling)
10. pipeline.py (pipeline of functions to be applied for modelling)
11. predictions.ipynb (all regression models here) 
12. preictions_as_binary_classification.ipynb (all classification models here)

Major data sources that required extensive data wrangling are placed in their own folders. Data was obtained, cleaned from these sources along with the target variable from FEWS and then merged in gather_data.ipynb before the models were run. 


# Authors
Annie Phan (aqp), Matthew Mauer (mrmauer), Nixon Joshy (nixon)
