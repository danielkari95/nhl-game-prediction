import numpy as np
import pandas as pd
import json
import measures
import os

from sklearn.preprocessing import StandardScaler

os.chdir('/home/daniel/Asiakirjat/GitHub/Gradu/')

def home_team_for(ids):
    if ids[0] == ids[1]:
        return 1
    else:
        return 0

def goal_difference(x):
    if 'OT' in x[0] or 'SO' in x[0]:
        return 0
    elif np.abs(x[1]-x[2]) <= 2:
        return x[1]-x[2]
    elif x[1]-x[2] > 2:
        return 3
    else:
        return -3

def result(result, binary=False):
    if not binary:
        if 'OT' in result or 'SO' in result:
            return 'b_tie'
        elif 'home' in result:
            return 'a_home_win'
        else:
            return 'c_away_win'
    else:
        if 'home' in result:
            return 'a_home_win'
        else:
            return 'b_away_win'

def save_files(X, y, X_columns, y_columns, suffix):
    np.save('data/numpy/X{0}.npy'.format(suffix), X)
    np.save('data/numpy/y{0}.npy'.format(suffix), y)
    with open('data/json/X{0}_columns.json'.format(suffix), 'w') as outfile:
        json.dump(X_columns, outfile)
    with open('data/json/y{0}_columns.json'.format(suffix), 'w') as outfile:
        json.dump(y_columns, outfile)
        
def load_data(suffix):
    X = np.load('data/numpy/X{0}.npy'.format(suffix))
    y = np.load('data/numpy/y{0}.npy'.format(suffix))
    with open('data/json/X{0}_columns.json'.format(suffix)) as jsonfile:
        X_columns = json.load(jsonfile)
    with open('data/json/y{0}_columns.json'.format(suffix)) as jsonfile:
        y_columns = json.load(jsonfile)
    return X, y, X_columns, y_columns

def to_3d_array(plays, zero_padding):
    n_games = len(plays.game_id.unique())
    if zero_padding:
        n_events = plays.groupby(['game_id']).size().describe()['max'].astype(np.int64)
    else:
        n_events = plays.groupby(['game_id']).size().describe()['min'].astype(np.int64)
    n_columns = len(plays.columns)

    X = np.zeros((n_games, n_events, n_columns)) # Initialize 3D array for data
    
    if zero_padding: 
        index_dummy = plays.columns.get_loc('event_dummy') 

    for i, game_id in enumerate(sorted(plays.game_id.unique())):
        game_events = plays[plays['game_id'] == game_id]
        if zero_padding: # Fill with dummy values
            fill = np.zeros((n_events-game_events.shape[0], n_columns))
            fill[:, index_dummy] = 1
            X[i] = np.vstack((game_events.to_numpy(), fill))
        # full_events = np.vstack((game_events, fill))
        else:
            X[i] = game_events.iloc[:n_events,:].to_numpy()
    assert np.isnan(X).any() == False  
    return X

def preprocess(plays=None, games=None, relevant_events=None, include=[], 
               save_as_file=False, return_arrays=False, zero_padding=True, 
               binary_y=False, y_goal_diff=False, suffix=""):    
    """ Function for preprocessing the play-by-play data for analysis. """
    if plays == None:
        plays = pd.read_csv('data/csv/game_plays.csv')

    if games == None:
        games = pd.read_csv('data/csv/game.csv')
    
    if relevant_events == None:
        relevant_events = [
                           'Faceoff', 
                           'Giveaway', 
                           'Blocked Shot', 
                           'Shot', 
                           'Stoppage',
                           'Hit', 
                           'Penalty', 
                           'Takeaway', 
                           'Missed Shot'
                           ]

    plays = plays[plays['event'].isin(relevant_events)] 
    # Only predict based on these events

    # plays['game_time'] = (plays['period']-1)*1200+plays['periodTime']
    plays = plays[plays['period'] <= 3] # Only plays in regulation included
    
    
    relevant_columns = ['game_id', 'team_id_for', 'team_id_against', 
                        'event', 'st_x', 'st_y']
    plays = plays.loc[:, relevant_columns]

    plays = plays.loc[~((plays['st_x']==0) & (plays['st_y']==0) & 
                    (plays['event']=='Faceoff'))]

    # print("Number of rows dropped:", plays.shape[0]-plays.dropna().shape[0])
    plays = plays.dropna()
    
    if not zero_padding:
        lower = plays.groupby(['game_id']).size().describe()['25%'].astype(np.int64)
        upper = plays.groupby(['game_id']).size().describe()['75%'].astype(np.int64)
        ids = []
        for game_id in plays['game_id'].unique():
            n_events = len(plays.loc[plays['game_id']==game_id])
            if n_events >= lower and n_events <= upper:
                ids.append(game_id)
        plays = plays.loc[plays['game_id'].isin(ids)]
    
    if 'distance' in include or 'danger' in include:
        plays['distance'] = plays[['st_x', 'st_y']].apply(measures.distance, axis=1)
    
    if 'distance' in include and 'angle' in include:
        plays['angle'] = plays[['st_x', 'distance']].apply(measures.angle, axis=1)
    
    if 'danger' in include:
        plays['danger'] = 'd_not_a_shot'
        shots = ['Shot', 'Missed Shot', 'Goal']
        plays.loc[plays['event'].isin(shots), 'danger'] = plays.loc[
                plays['event'].isin(shots), 'distance'].apply(measures.danger)
        plays = pd.get_dummies(plays, columns=['event', 'danger'])
    else:
        plays = pd.get_dummies(plays, columns=['event'])
    
    if zero_padding:
        plays['event_dummy'] = 0

    plays.rename(columns=lambda x: x.replace(' ', '_').lower(), inplace=True)

    games = games.loc[games['game_id'].isin(plays['game_id'].unique()), :]
    games = games.sort_values(by='game_id')
    games = games.loc[:, ['game_id', 'home_team_id', 'outcome', 'home_goals', 'away_goals']]

    plays = plays.merge(games, on='game_id')

    plays['home_team_for'] = plays[['team_id_for', 'home_team_id']].apply(
                     home_team_for, axis=1)
    
    if not y_goal_diff:
        games['y'] = games['outcome'].apply(result, binary=binary_y)
    else:
        games['y'] = games[['outcome', 'home_goals', 
                                     'away_goals']].apply(goal_difference, axis=1)
    
    y = pd.get_dummies(games['y'])
    y_columns = dict(zip(y.columns, range(len(y.columns))))
    y = y.to_numpy()
    
    if 'zones' in include:
        plays['offensive_zone_home'] = plays[['st_x', 'home_team_for']].apply(
                                measures.offensive_zone_home, axis=1)
        plays['offensive_zone_away'] = plays[['st_x', 'home_team_for']].apply(
                                measures.offensive_zone_away, axis=1)

    predict_columns = ['game_id', 'st_x', 'st_y', 'distance', 'angle',
                        'home_team_for', 'offensive_zone_home', 'offensive_zone_away']

    plays = plays.loc[:, (plays.columns.isin(predict_columns) | 
            plays.columns.str.contains('event') |
            plays.columns.str.contains('danger'))]


    
    scale_columns = ['st_x', 'st_y', 'distance', 'angle']
    plays.loc[:, plays.columns.isin(scale_columns)] = StandardScaler().fit_transform(
                            plays.loc[:, plays.columns.isin(scale_columns)])
        
    if 'distance' not in include and 'danger' in include:
        plays = plays.drop(columns=['distance'])

    X = to_3d_array(plays, zero_padding)  
    X_columns = dict(zip(plays.columns, range(len(plays.columns))))
    
    if save_as_file:
        save_files(X, y, X_columns, y_columns, suffix)
    
    if return_arrays:
        return X, y, X_columns, y_columns