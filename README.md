# Predicting hunger and food insecurity

## Summary

Hunger and food insecurity has risen around the world over the last five years pushing many into poverty traps and limiting their human potential. The project aimed to use Machine Learning to predict future outbreaks of famines and incidences of food insecurity focusing on countries in the Horn of Africa. Weather, agricultural food prices, conflict and demographic data were chosen as the likely contributing factors to predicting hunger based on literature review. These datasets were then collected, cleaned and merged with the hunger scores based on month, year and geographic location. We approached the question using both prediction and classification models. We first attempted to predict the hunger score using multiple regression models and then proceeded to discretize the target to test different classification models that classified regions for a given time period as food insecure or not.  Based on multiple model evaluations, conflicts especially battles was a major predictor of  food insecurity followed by food prices and population. The results were satisfactory but there needs to be a more thorough analysis with a much larger dataset to reliably predict hunger especially if it is to be used to direct multi-million dollar food aid supply chains.

## Order of Execution

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

Geographic and target variable data were download to local disc to be worked. Both such datasets may be aquired at [FEWS NET](https://fews.net/fews-data/333).


# Authors
Annie Phan (aqp), Matthew Mauer (mrmauer), Nixon Joshy (nixon)
