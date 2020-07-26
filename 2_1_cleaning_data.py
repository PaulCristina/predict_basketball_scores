#------------------------------------------------------------------------------------------------------------#
### 1.0 LOAD LIBRARY'S ----
import os, sys
import pandas as pd
import numpy as np
import dateutil
import re

# Workflow
# 1. Take the Teams and Games CSV files and load them into their own DataFrames.
# Don’t forget to include their column names to make everything easier later on.
# 2. Clean the data.
# Cleaning includes making sure all of the team names match 
# (Ex. St Joe vs St. Joe) between the Teams and Games DataFrames.
# 3. Prepare the data for processing by creating columns that
#  we can train against.
# Get creative. Try and think of data points that make sense with 
# respect to basketball. For example, home court is a very real thing.
#  What can you create that will let the network use that to learn?
# 4. Save the output from your added columns to 
# Games-Calculated.csv.

### set header for datasets
columns_games_df = ['date','away','away_points','home','home_points']
columns_teams_df = ['conference','college']

### change `DATA_DIR` to the location where movielens-20m dataset sits
DATA_DIR = 'C:/Users/Guest01/Documents/github_projects/predict_basketball_scores/data'
games = pd.read_csv(os.path.join(DATA_DIR, 'Games.csv'), names = columns_games_df)
teams = pd.read_csv(os.path.join(DATA_DIR, 'Teams.csv'), names = columns_teams_df)

games.head()
teams.head()

### clean strings
games['away'] = games['away'].str.replace('&amp;', '&')
games['home'] = games['home'].str.replace('&amp;', '&')

games['away'] = games['away'].str.replace('&#039;', "'")
games['home'] = games['home'].str.replace('&#039;', "'")

games['away'] = games['away'].str.replace('.', "")
games['home'] = games['home'].str.replace('.;', "")

