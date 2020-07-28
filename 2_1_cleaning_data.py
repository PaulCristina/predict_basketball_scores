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
columns_games_df = ['Date','HomeTeam','HomeScore','AwayTeam','AwayScore']
columns_teams_df = ['Conference','Team']

### change `DATA_DIR` to the location where movielens-20m dataset sits
games_csv = 'https://liveproject-resources.s3.amazonaws.com/other/deeplearningbasketballscores/Games.csv'
games = pd.read_csv(games_csv, header=None, names=columns_games_df)

teams_csv = 'https://liveproject-resources.s3.amazonaws.com/other/deeplearningbasketballscores/Teams.csv'
teams = pd.read_csv(teams_csv, header=None, names=columns_teams_df)


games.head()
teams.head()

### remove any games without a score
games = games.drop( games[(games['HomeScore'] == 0)].index )
games = games.drop( games[(games['AwayScore'] == 0)].index )

# ### clean strings
# games['HomeTeam'] = games['HomeTeam'].str.replace('&amp;', '&')
# games['AwayTeam'] = games['AwayTeam'].str.replace('&amp;', '&')

# games['HomeTeam'] = games['HomeTeam'].str.replace('&#039;', "'")
# games['AwayTeam'] = games['AwayTeam'].str.replace('&#039;', "'")

### break the Games DataFrame into seasons
s2015 = games[(games['Date'] > '2015-11-01') & (games['Date'] < '2016-04-15')].copy()
s2016 = games[(games['Date'] > '2016-11-01') & (games['Date'] < '2017-04-15')].copy()
s2017 = games[(games['Date'] > '2017-11-01') & (games['Date'] < '2018-04-15')].copy()
s2018 = games[(games['Date'] > '2018-11-01') & (games['Date'] < '2019-04-15')].copy()

print('2015 (%s): %s - %s' % (s2015.shape[0],np.min(s2015.Date),np.max(s2015.Date)))
print('2016 (%s): %s - %s' % (s2016.shape[0],np.min(s2016.Date),np.max(s2016.Date)))
print('2017 (%s): %s - %s' % (s2017.shape[0],np.min(s2017.Date),np.max(s2017.Date)))
print('2018 (%s): %s - %s' % (s2018.shape[0],np.min(s2018.Date),np.max(s2018.Date)))

### Clean the team names if they don't match the Teams entry


# games['away'] = games['away'].str.replace('.', "")
# games['home'] = games['home'].str.replace('.;', "")

def RenameTeams(df_games, column_name):
  df_games.loc[ df_games[column_name] == 'A&M-Corpus Chris'		, column_name ] = 		'Texas A&M-CC'	
  df_games.loc[ df_games[column_name] == 'Alabama St.'		, column_name ] = 		'Alabama State'		
  df_games.loc[ df_games[column_name] == 'Albany (NY)'		, column_name ] = 		'Albany'				
  df_games.loc[ df_games[column_name] == 'Alcorn St.'			, column_name ] = 		'Alcorn State'		
  df_games.loc[ df_games[column_name] == 'American'			, column_name ] = 		'American University'
  df_games.loc[ df_games[column_name] == 'Appalachian St.'			, column_name ] = 		'Appalachian State'	
  df_games.loc[ df_games[column_name] == 'Arizona St.'		, column_name ] = 		'Arizona State'						
  df_games.loc[ df_games[column_name] == 'Army West Point'		, column_name ] = 		'Army'					
  df_games.loc[ df_games[column_name] == 'Ark.-Pine Bluff'		, column_name ] = 		'Arkansas-Pine Bluff'
  df_games.loc[ df_games[column_name] == 'UALR'				, column_name ] = 		'Arkansas-Little Rock'	
  df_games.loc[ df_games[column_name] == 'Little Rock'				, column_name ] = 		'Arkansas-Little Rock'			
  df_games.loc[ df_games[column_name] == 'Arkansas St.'		, column_name ] = 		'Arkansas State'		
  df_games.loc[ df_games[column_name] == 'Ball St.'			, column_name ] = 		'Ball State'			
  df_games.loc[ df_games[column_name] == 'Boise St.'			, column_name ] = 		'Boise State'		
  df_games.loc[ df_games[column_name] == 'Boston U.'			, column_name ] = 		'Boston University'			
  df_games.loc[ df_games[column_name] == 'Cal Baptist'	, column_name ] = 		'California Baptist'			
  df_games.loc[ df_games[column_name] == 'Charleston So.'	, column_name ] = 		'Charleston Southern'			
  df_games.loc[ df_games[column_name] == 'Cent. Conn. St.'	, column_name ] = 		'Central Connecticut State'	
  df_games.loc[ df_games[column_name] == 'Central Conn. St.'	, column_name ] = 		'Central Connecticut State'	
  df_games.loc[ df_games[column_name] == 'Central Mich.'	, column_name ] = 		'Central Michigan'	
  df_games.loc[ df_games[column_name] == 'Col. of Charleston'	, column_name ] = 		'Charleston'			
  df_games.loc[ df_games[column_name] == 'Chicago St.'		, column_name ] = 		'Chicago State'		
  df_games.loc[ df_games[column_name] == 'Cleveland St.'		, column_name ] = 		'Cleveland State'		
  df_games.loc[ df_games[column_name] == 'Coastal Caro.'		, column_name ] = 		'Coastal Carolina'				
  df_games.loc[ df_games[column_name] == 'Colorado St.'		, column_name ] = 		'Colorado State'	
  df_games.loc[ df_games[column_name] == 'Coppin St.'			, column_name ] = 		'Coppin State'			
  df_games.loc[ df_games[column_name] == 'Bakersfield'		, column_name ] = 		'Cal State Bakersfield'	
  df_games.loc[ df_games[column_name] == 'CSU Bakersfield'		, column_name ] = 		'Cal State Bakersfield'		
  df_games.loc[ df_games[column_name] == 'Bryant'		, column_name ] = 		'Bryant University'	
  df_games.loc[ df_games[column_name] == 'Cal St. Fullerton'	, column_name ] = 		'Cal State Fullerton'
  df_games.loc[ df_games[column_name] == 'CSU Fullerton'	, column_name ] = 		'Cal State Fullerton'		
  df_games.loc[ df_games[column_name] == 'CSUN'	, column_name ] = 		'Cal State Northridge'	
  df_games.loc[ df_games[column_name] == 'Cal St. Northridge'	, column_name ] = 		'Cal State Northridge'						
  df_games.loc[ df_games[column_name] == 'Central Ark.'		, column_name ] = 		'Central Arkansas'						
  df_games.loc[ df_games[column_name] == 'Delaware St.'		, column_name ] = 		'Delaware State'		
  df_games.loc[ df_games[column_name] == 'Detroit'			, column_name ] = 		'Detroit Mercy'		
  df_games.loc[ df_games[column_name] == 'East Tenn. St.'		, column_name ] = 		'East Tennessee State'
  df_games.loc[ df_games[column_name] == 'Eastern Ill.'		, column_name ] = 		'Eastern Illinois'		
  df_games.loc[ df_games[column_name] == 'Eastern Ky.'		, column_name ] = 		'Eastern Kentucky'		
  df_games.loc[ df_games[column_name] == 'Eastern Mich.'		, column_name ] = 		'Eastern Michigan'	
  df_games.loc[ df_games[column_name] == 'Eastern Wash.'		, column_name ] = 		'Eastern Washington'
  df_games.loc[ df_games[column_name] == "Fairleigh D'son"		, column_name ] = 		'Fairleigh Dickinson'				
  df_games.loc[ df_games[column_name] == 'FGCU'		, column_name ] = 		'Florida Gulf Coast'						
  df_games.loc[ df_games[column_name] == 'FIU'				, column_name ] = 		'Florida International'					
  df_games.loc[ df_games[column_name] == 'Fla. Atlantic'		, column_name ] = 		'Florida Atlantic'
  df_games.loc[ df_games[column_name] == 'Florida St.'		, column_name ] = 		'Florida State'			
  df_games.loc[ df_games[column_name] == 'Fresno St.'			, column_name ] = 		'Fresno State'		
  df_games.loc[ df_games[column_name] == 'Fort Wayne'		, column_name ] = 		'Purdue Fort Wayne'		
  df_games.loc[ df_games[column_name] == 'IPFW'				, column_name ] = 		'Purdue Fort Wayne'				
  df_games.loc[ df_games[column_name] == 'Ga. Southern'		, column_name ] = 		'Georgia Southern'			
  df_games.loc[ df_games[column_name] == 'Georgia St.'		, column_name ] = 		'Georgia State'			
  df_games.loc[ df_games[column_name] == 'Geo. Washington'		, column_name ] = 		'George Washington'				
  df_games.loc[ df_games[column_name] == 'Grambling'		, column_name ] = 		'Grambling State'		
  df_games.loc[ df_games[column_name] == 'Humboldt St.'		, column_name ] = 		'Humboldt State'		
  df_games.loc[ df_games[column_name] == 'Idaho St.'			, column_name ] = 		'Idaho State'			
  df_games.loc[ df_games[column_name] == 'Illinois St.'		, column_name ] = 		'Illinois State'		
  df_games.loc[ df_games[column_name] == 'Iowa St.'			, column_name ] = 		'Iowa State'			
  df_games.loc[ df_games[column_name] == 'Indiana St.'		, column_name ] = 		'Indiana State'		
  df_games.loc[ df_games[column_name] == 'Jackson St.'		, column_name ] = 		'Jackson State'		
  df_games.loc[ df_games[column_name] == 'Jacksonville St.'		, column_name ] = 		'Jacksonville State'			
  df_games.loc[ df_games[column_name] == 'Kansas St.'		, column_name ] = 		'Kansas State'
  df_games.loc[ df_games[column_name] == 'Kennesaw St.'		, column_name ] = 		'Kennesaw State'		
  df_games.loc[ df_games[column_name] == 'Kent St.'			, column_name ] = 		'Kent State'			
  df_games.loc[ df_games[column_name] == 'Louisiana'		, column_name ] = 		'Louisiana-Lafayette'
  df_games.loc[ df_games[column_name] == 'Lamar University'		, column_name ] = 		'Lamar'	
  df_games.loc[ df_games[column_name] == 'La.-Monroe'		, column_name ] = 		'Louisiana-Monroe'		
  df_games.loc[ df_games[column_name] == 'Long Beach St.'		, column_name ] = 		'Long Beach State'	
  df_games.loc[ df_games[column_name] == 'Long Island'		, column_name ] = 		'LIU Brooklyn'
  df_games.loc[ df_games[column_name] == 'LMU'	, column_name ] = 		'Loyola Marymount'					
  df_games.loc[ df_games[column_name] == 'Loyola Chicago'	, column_name ] = 		'Loyola (IL)'			
  df_games.loc[ df_games[column_name] == 'Loyola Maryland'	, column_name ] = 		'Loyola (MD)'			
  df_games.loc[ df_games[column_name] == 'Loyola (Md.)'	, column_name ] = 		'Loyola (MD)'		
  df_games.loc[ df_games[column_name] == 'UMES'		, column_name ] = 		'Maryland-Eastern Shore'
  df_games.loc[ df_games[column_name] == 'Miami (Fla.)'		, column_name ] = 		'Miami (FL)'
  df_games.loc[ df_games[column_name] == 'Miami (Ohio)'		, column_name ] = 		'Miami (OH)'
  df_games.loc[ df_games[column_name] == "Mt. St. Mary's"		, column_name ] = 		"Mount St Mary's"			
  df_games.loc[ df_games[column_name] == 'Mass.-Lowell'		, column_name ] = 		'Massachusetts-Lowell'				
  df_games.loc[ df_games[column_name] == 'McNeese'		, column_name ] = 		'McNeese State'										
  df_games.loc[ df_games[column_name] == 'McNeese '		, column_name ] = 		'McNeese State'						
  df_games.loc[ df_games[column_name] == 'McNeese St.'		, column_name ] = 		'McNeese State'			
  df_games.loc[ df_games[column_name] == 'Middle Tenn.'		, column_name ] = 		'Middle Tennessee'		
  df_games.loc[ df_games[column_name] == 'Mississippi St.'		, column_name ] = 		'Mississippi State'	
  df_games.loc[ df_games[column_name] == 'Mississippi Val.'		, column_name ] = 		'Mississippi Valley State'			
  df_games.loc[ df_games[column_name] == 'Mich. St. '		, column_name ] = 		'Michigan State'		
  df_games.loc[ df_games[column_name] == 'Michigan St.'		, column_name ] = 		'Michigan State'			
  df_games.loc[ df_games[column_name] == 'Mississippi'			, column_name ] = 		'Ole Miss'			
  df_games.loc[ df_games[column_name] == 'Missouri St.'		, column_name ] = 		'Missouri State'	
  df_games.loc[ df_games[column_name] == 'Montana St.'		, column_name ] = 		'Montana State'		
  df_games.loc[ df_games[column_name] == 'Morehead St.'		, column_name ] = 		'Morehead State'		
  df_games.loc[ df_games[column_name] == 'Morgan St.'			, column_name ] = 		'Morgan State'			
  df_games.loc[ df_games[column_name] == 'Murray St.'			, column_name ] = 		'Murray State'		
  df_games.loc[ df_games[column_name] == 'N.C. A&T'	, column_name ] = 		'North Carolina A&T'
  df_games.loc[ df_games[column_name] == 'N.C. Central'	, column_name ] = 		'North Carolina Central'
  df_games.loc[ df_games[column_name] == 'New Mexico St.'	, column_name ] = 		'New Mexico State'				
  df_games.loc[ df_games[column_name] == 'NC State'	, column_name ] = 		'North Carolina State'
  df_games.loc[ df_games[column_name] == 'North Carolina St.'	, column_name ] = 		'North Carolina State'	
  df_games.loc[ df_games[column_name] == 'North Dakota St.'	, column_name ] = 		'North Dakota State'				
  df_games.loc[ df_games[column_name] == 'Northern Ariz.'	, column_name ] = 		'Northern Arizona'				
  df_games.loc[ df_games[column_name] == 'Northern Colo.'	, column_name ] = 		'Northern Colorado'		
  df_games.loc[ df_games[column_name] == 'Northern Ill.'	, column_name ] = 		'Northern Illinois'			
  df_games.loc[ df_games[column_name] == "N'western St."	, column_name ] = 		"Northwestern State"	
  df_games.loc[ df_games[column_name] == 'Northwestern St.'	, column_name ] = 		"Northwestern State"				
  df_games.loc[ df_games[column_name] == 'Nicholls St.'		, column_name ] = 		'Nicholls State'		
  df_games.loc[ df_games[column_name] == 'Norfolk St.'		, column_name ] = 		'Norfolk State'		
  df_games.loc[ df_games[column_name] == 'Northern Ky.'		, column_name ] = 		'Northern Kentucky'	
  df_games.loc[ df_games[column_name] == 'Ohio St.'			, column_name ] = 		'Ohio State'			
  df_games.loc[ df_games[column_name] == 'Ohio St. '			, column_name ] = 		'Ohio State'			
  df_games.loc[ df_games[column_name] == 'Oklahoma St.'		, column_name ] = 		'Oklahoma State'	
  df_games.loc[ df_games[column_name] == 'Oregon St.'		, column_name ] = 		'Oregon State'	
  df_games.loc[ df_games[column_name] == 'Neb. Omaha'			, column_name ] = 		'Nebraska-Omaha'	
  df_games.loc[ df_games[column_name] == 'Omaha'			, column_name ] = 		'Nebraska-Omaha'					
  df_games.loc[ df_games[column_name] == 'Penn'			, column_name ] = 		'Pennsylvania'								
  df_games.loc[ df_games[column_name] == 'Penn St.'			, column_name ] = 		'Penn State'				
  df_games.loc[ df_games[column_name] == 'Prairie View'		, column_name ] = 		'Prairie View A&M'		
  df_games.loc[ df_games[column_name] == 'Portland St.'		, column_name ] = 		'Portland State'		
  df_games.loc[ df_games[column_name] == 'S.C. Upstate'	, column_name ] = 		'USC Upstate'	
  df_games.loc[ df_games[column_name] == 'S. Carolina St.'	, column_name ] = 		'South Carolina State'
  df_games.loc[ df_games[column_name] == 'South Carolina St.'	, column_name ] = 		'South Carolina State'			
  df_games.loc[ df_games[column_name] == 'Sacramento St.'		, column_name ] = 		'Sacramento State'			
  df_games.loc[ df_games[column_name] == 'Sam Houston St.'		, column_name ] = 		'Sam Houston State'				
  df_games.loc[ df_games[column_name] == 'San Diego St.'		, column_name ] = 		'San Diego State'		
  df_games.loc[ df_games[column_name] == 'San Jose St.'		, column_name ] = 		'San Jose State'		
  df_games.loc[ df_games[column_name] == 'Savannah St.'		, column_name ] = 		'Savannah State'		
  df_games.loc[ df_games[column_name] == 'Seattle U'			, column_name ] = 		'Seattle'			
  df_games.loc[ df_games[column_name] == 'SFA'	, column_name ] = 		'Stephen F Austin'
  df_games.loc[ df_games[column_name] == 'Stephen F. Austin'	, column_name ] = 		'Stephen F Austin'					
  df_games.loc[ df_games[column_name] == 'SIU Edwardsville'	, column_name ] = 		'SIU-Edwardsville'						
  df_games.loc[ df_games[column_name] == 'SIUE'	, column_name ] = 		'SIU-Edwardsville'					
  df_games.loc[ df_games[column_name] == 'South Ala.'			, column_name ] = 		'South Alabama'			
  df_games.loc[ df_games[column_name] == 'South Dakota St.'			, column_name ] = 		'South Dakota State'	
  df_games.loc[ df_games[column_name] == 'South Fla.'			, column_name ] = 		'South Florida'		
  df_games.loc[ df_games[column_name] == 'Southern Ill.'		, column_name ] = 		'Southern Illinois'	
  df_games.loc[ df_games[column_name] == 'Southeast Mo. St.'		, column_name ] = 		'Southeast Missouri State'
  df_games.loc[ df_games[column_name] == 'Southeastern La.'		, column_name ] = 		'Southeastern Louisiana'		
  df_games.loc[ df_games[column_name] == 'Southern Miss.'		, column_name ] = 		'Southern Miss'		
  df_games.loc[ df_games[column_name] == 'Southern U.'		, column_name ] = 		'Southern University'	
  df_games.loc[ df_games[column_name] == 'Southern Univ.'		, column_name ] = 		'Southern University'	
  df_games.loc[ df_games[column_name] == "St. Bonaventure"	, column_name ] = 		"St Bonaventure"			
  df_games.loc[ df_games[column_name] == "St. Francis (B'klyn)"	, column_name ] = 		"St Francis (BKN)"			
  df_games.loc[ df_games[column_name] == 'St. Francis (NY)'	, column_name ] = 		"St Francis (BKN)"
  df_games.loc[ df_games[column_name] == 'St. Francis (PA)'	, column_name ] = 		"St Francis (PA)"	
  df_games.loc[ df_games[column_name] == 'St. Francis (Pa.)'	, column_name ] = 		"St Francis (PA)"	
  df_games.loc[ df_games[column_name] == "Saint Joseph's"	, column_name ] = 		"Saint Joseph's (PA)"
  df_games.loc[ df_games[column_name] == "St. Mary's (CA)"	, column_name ] = 		"Saint Mary's"	
  df_games.loc[ df_games[column_name] == "St. Mary's (Cal.)"	, column_name ] = 		"Saint Mary's"	
  df_games.loc[ df_games[column_name] == "St. Peter's"	, column_name ] = 		"St Peter's"	
  df_games.loc[ df_games[column_name] == "St. John's (NY)"	, column_name ] = 		"St John's"		
  df_games.loc[ df_games[column_name] == "St. John's "	, column_name ] = 		"St John's"				
  df_games.loc[ df_games[column_name] == 'Tennessee St.'		, column_name ] = 		'Tennessee State'		
  df_games.loc[ df_games[column_name] == 'Texas A&M-C.C.'			, column_name ] = 		'Texas A&M-CC'			
  df_games.loc[ df_games[column_name] == 'Texas St.'			, column_name ] = 		'Texas State'			
  df_games.loc[ df_games[column_name] == 'UC Santa Barbara'	, column_name ] = 		'UC Santa Barb.'		
  df_games.loc[ df_games[column_name] == 'Ill.-Chicago'		, column_name ] = 		'UIC'					
  df_games.loc[ df_games[column_name] == 'Md.-East. Shore'	, column_name ] = 		'Maryland-Eastern Shore'					
  df_games.loc[ df_games[column_name] == 'UNCG'		, column_name ] = 		'UNC Greensboro'					
  df_games.loc[ df_games[column_name] == 'UNCW'		, column_name ] = 		'North Carolina-Wilmington'					
  df_games.loc[ df_games[column_name] == 'UNC Wilmington'		, column_name ] = 		'North Carolina-Wilmington'					
  df_games.loc[ df_games[column_name] == 'Southern California', column_name ] = 		'USC'					
  df_games.loc[ df_games[column_name] == 'UConn'			, column_name ] = 		'Connecticut'					
  df_games.loc[ df_games[column_name] == 'UC Santa Barb.'			, column_name ] = 		'UC Santa Barbara'						
  df_games.loc[ df_games[column_name] == 'UIC'			, column_name ] = 		'Illinois-Chicago'						
  df_games.loc[ df_games[column_name] == 'UNI'			, column_name ] = 		'Northern Iowa'							
  df_games.loc[ df_games[column_name] == 'UT Arlington'			, column_name ] = 		'Texas-Arlington'		
  df_games.loc[ df_games[column_name] == 'UT Arlington '			, column_name ] = 		'Texas-Arlington'		
  df_games.loc[ df_games[column_name] == 'UT Martin'			, column_name ] = 		'Tennessee-Martin'		
  df_games.loc[ df_games[column_name] == 'UTRGV'			, column_name ] = 		'Texas Rio Grande Valley'		
  df_games.loc[ df_games[column_name] == 'Utah St.'			, column_name ] = 		'Utah State'				
  df_games.loc[ df_games[column_name] == 'VCU'			, column_name ] = 		'Virginia Commonwealth'		
  df_games.loc[ df_games[column_name] == 'VMI'			, column_name ] = 		'Virginia Military'		
  df_games.loc[ df_games[column_name] == 'Washington St.'			, column_name ] = 		'Washington State'			
  df_games.loc[ df_games[column_name] == 'Weber St.'			, column_name ] = 		'Weber State'			
  df_games.loc[ df_games[column_name] == 'Western Caro.'		, column_name ] = 		'Western Carolina'
  df_games.loc[ df_games[column_name] == 'Western Ill.'		, column_name ] = 		'Western Illinois'	
  df_games.loc[ df_games[column_name] == 'Western Ky.'		, column_name ] = 		'Western Kentucky'	
  df_games.loc[ df_games[column_name] == 'Western Mich.'		, column_name ] = 		'Western Michigan'		
  df_games.loc[ df_games[column_name] == 'Wichita St.'		, column_name ] = 		'Wichita State'		
  df_games.loc[ df_games[column_name] == 'Wright St.'			, column_name ] = 		'Wright State'	
  df_games.loc[ df_games[column_name] == 'Youngstown St.'			, column_name ] = 		'Youngstown State'
  
RenameTeams(s2015,'HomeTeam')
RenameTeams(s2015,'AwayTeam')
RenameTeams(s2016,'HomeTeam')
RenameTeams(s2016,'AwayTeam')
RenameTeams(s2017,'HomeTeam')
RenameTeams(s2017,'AwayTeam')
RenameTeams(s2018,'HomeTeam')
RenameTeams(s2018,'AwayTeam')  

### calculate different statistics
# team's scoring average at home
# team's scoring average away
# team's home defensive average
# team's away defensive average
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.expanding.html
# expanding().mean() Documentation gives us a rolling average. 

def CalculateScores(df):
  for index, row in teams.iterrows():
    ## Home Team Stats
    df.loc[ df['HomeTeam'] == row.Team, 'HomeScoreAverage' ] = df[(df['HomeTeam'] == row.Team)]['HomeScore'].expanding().mean()
    df.loc[ df['HomeTeam'] == row.Team, 'HomeDefenseAverage' ] = df[(df['HomeTeam'] == row.Team)]['AwayScore'].expanding().mean()
    ## Away Team Stats
    df.loc[ df['AwayTeam'] == row.Team, 'AwayScoreAverage' ] = df[(df['AwayTeam'] == row.Team)]['AwayScore'].expanding().mean()
    df.loc[ df['AwayTeam'] == row.Team, 'AwayDefenseAverage' ] = df[(df['AwayTeam'] == row.Team)]['HomeScore'].expanding().mean()
    
    
CalculateScores(s2015)
CalculateScores(s2016)
CalculateScores(s2017)
CalculateScores(s2018)    

### Combine the data, remove NA values, and calculate the results
all_data = pd.concat([s2015,s2016,s2017,s2018])
#Remove NAN values
print(len(all_data))
all_data.dropna(inplace=True)
print(len(all_data))

all_data['Result'] = games.HomeScore - games.AwayScore

# Save both DataFrames to csv files
all_data.to_csv('C:/Users/Guest01/Documents/github_projects/predict_basketball_scores/data/all_data.csv')
