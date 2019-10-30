import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler

def distance(x):
    return np.sqrt((x[0]-90.0)**2+x[1]**2)

def angle(x):
    if 90.0-x[0] == x[1]:
        return 0
    else:
        return np.degrees(np.arccos((90.0-x[0])/x[1]))
    
def home_team_for(ids):
    if ids[0] == ids[1]:
        return 1
    else:
        return 0

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

def danger(distance):
    if distance <= 15:
        return 'a_high'
    elif distance <= 30:
        return 'b_med'
    else:
        return 'c_low'

def offensive_zone_home(x):
    if x[0] >= 25 and x[1] == 1:
        return 1
    elif x[0] <= -25 and x[1] == 0:
        return 1
    else:
        return 0

def offensive_zone_away(x):
    if x[0] >= 25 and x[1] == 0:
        return 1
    if x[0] <= -25 and x[1] == 1:
        return 1
    else:
        return 0    

def load_data(plays, games, relevant_events=None, save_as_file=False, 
              return_arrays=False, include=[], zero_padding=True, 
              binary_y=False, suffix=""):    
    """ Function for loading the play-by-play data and preprocessing it. """
    
    if relevant_events == None:
        relevant_events = [
                           'Faceoff', 
                           'Giveaway', 
                           'Blocked Shot', 
                           'Shot', 
                           'Hit', 
                           'Penalty', 
                           'Takeaway', 
                           'Missed Shot'
                           ]

    plays = plays[plays['event'].isin(relevant_events)] # Only predict based on these events

    # plays['game_time'] = (plays['period']-1)*1200+plays['periodTime']
    plays = plays[plays['period'] <= 3] # Only plays in regulation included
    
    
    relevant_columns = ['play_id', 'game_id', 'team_id_for', 'team_id_against', 
                        'event', 'st_x', 'st_y']
    plays = plays.loc[:, relevant_columns]

    plays = plays.loc[~((plays['st_x']==0) & (plays['st_y']==0) & 
                    (plays['event']=='Faceoff'))]

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
        plays['distance'] = plays[['st_x', 'st_y']].apply(distance, axis=1)
    
    if 'distance' in include and 'angle' in include:
        plays['angle'] = plays[['st_x', 'distance']].apply(angle, axis=1)
    
    if 'danger' in include:
        plays['danger'] = 'd_not_a_shot'
        shots = ['Shot', 'Missed Shot', 'Goal']
        plays.loc[plays['event'].isin(shots), 'danger'] = plays.loc[
                plays['event'].isin(shots), 'distance'].apply(danger)
        plays = pd.get_dummies(plays, columns=['event', 'danger'])
        
    else:
        plays = pd.get_dummies(plays, columns=['event'])
    
    if zero_padding:
        plays['event_dummy'] = 0

    plays.rename(columns=lambda x: x.replace(' ', '_').lower(), inplace=True)

    games = games.loc[games['game_id'].isin(plays['game_id'].unique()), :]
    games = games.sort_values(by='game_id')

    plays = plays.merge(games, on='game_id')

    plays['home_team_for'] = plays[['team_id_for', 'home_team_id']].apply(
                     home_team_for, axis=1)
    
    games['result'] = games['outcome'].apply(result, binary=binary_y)
    y = pd.get_dummies(games['result']).to_numpy(dtype=np.int8)
    
    if 'zones' in include:
        plays['offensive_zone_home'] = plays[['st_x', 'home_team_for']].apply(
                                offensive_zone_home, axis=1)
        plays['offensive_zone_away'] = plays[['st_x', 'home_team_for']].apply(
                                offensive_zone_away, axis=1)

    relevant_columns = ['game_id', 'st_x', 'st_y', 'distance', 'angle',
                        'home_team_for', 'offensive_zone_home', 'offensive_zone_away']

    plays = plays.loc[:, (plays.columns.isin(relevant_columns) | 
            plays.columns.str.contains('event') |
            plays.columns.str.contains('danger'))]


    if ('st_x' or 'st_y' or 'distance' or 'angle') in include:
        scale_columns = ['st_x', 'st_y', 'distance', 'angle']
        plays.loc[:, plays.columns.isin(scale_columns)] = StandardScaler().fit_transform(
                            plays.loc[:, plays.columns.isin(scale_columns)])

    else:
        plays = plays.drop(columns=['st_x', 'st_y'])
        
    if 'distance' not in include and 'danger' in include:
        plays = plays.drop(columns=['distance'])

    n_games = len(plays.game_id.unique())
    n_events = plays.groupby(['game_id']).size().describe()['min'].astype(np.int64)
    if zero_padding:
        n_events = plays.groupby(['game_id']).size().describe()['max'].astype(np.int64)
    n_columns = len(plays.columns)-1

    X = np.zeros((n_games, n_events, n_columns)) # Initialize 3D arrays for data
    
    if zero_padding:
        index_dummy = plays.columns.get_loc('event_dummy')-1

    for i, game_id in enumerate(sorted(plays.game_id.unique())):
        game_events = plays[plays['game_id'] == game_id].iloc[:,1:]
        if zero_padding:
            fill = np.zeros((n_events-game_events.shape[0], n_columns))
            fill[:, index_dummy] = 1
            X[i] = np.vstack((game_events.to_numpy(), fill))
        # full_events = np.vstack((game_events, fill))
        else:
            X[i] = game_events.iloc[:n_events,:].to_numpy()    

    # np.save('X_full_features.npy', X)
    # np.save('X_without_coords.npy', X)

    if save_as_file:
        np.save('data/numpy/X{0}.npy'.format(suffix), X)
        np.save('data/numpy/y{0}.npy'.format(suffix), y)
    
    if return_arrays:
        return X, y
