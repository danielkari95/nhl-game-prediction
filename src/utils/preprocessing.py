import numpy as np
import pandas as pd
import json
import measures
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def home_team_for(ids):
    if ids[0] == ids[1]:
        return 1
    else:
        return 0

def goal_difference(x):
    if 'OT' in x[0] or 'SO' in x[0]:
        return 0
    else:
        return x[1]-x[2]

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

def save_data(data_train, data_test, columns, suffix, suffix_y="", X=False):

    name = 'X' if X else 'y'
    
    cd = os.getcwd()

    if not os.path.exists('data/{0}'.format(suffix)):
        os.mkdir('data/{0}'.format(suffix))
    
    os.chdir('data/{0}'.format(suffix))
    
    if not os.path.exists('numpy'):
        os.mkdir('numpy')
    
    if not os.path.exists('json'):
        os.mkdir('json')
    
    np.save('numpy/{0}_train{1}.npy'.format(name, suffix_y), data_train)
    np.save('numpy/{0}_test{1}.npy'.format(name, suffix_y), data_test)
    
    with open('json/{0}_columns{1}.json'.format(name, suffix_y), 'w') as outfile:
        json.dump(columns, outfile)

    os.chdir(cd)
        
def load_data(suffix, y_goal_diff=False):

    name = '_goal_diff' if y_goal_diff else '_result'

    cd = os.getcwd()
    os.chdir('data/{0}'.format(suffix))
    
    X_train = np.load('numpy/X_train.npy')
    X_test = np.load('numpy/X_test.npy')
    y_train = np.load('numpy/y_train{0}.npy'.format(name))
    y_test = np.load('numpy/y_test{0}.npy'.format(name))
    
    with open('json/X_columns.json') as jsonfile:
        X_columns = json.load(jsonfile)
    with open('json/y_columns{0}.json'.format(name)) as jsonfile:
        y_columns = json.load(jsonfile)
    
    os.chdir(cd)
    
    return X_train, X_test, y_train, y_test, X_columns, y_columns

def to_3d_array(plays):
    n_games = len(plays.game_id.unique())
    n_events = plays.groupby(['game_id']).size().describe()['max'].astype(np.int64)
    n_columns = len(plays.columns)

    X = np.zeros((n_games, n_events, n_columns)) # Initialize 3D array for data
    
    index_dummy = plays.columns.get_loc('event_dummy') 

    for i, game_id in enumerate(sorted(plays.game_id.unique())):
        game_events = plays[plays['game_id'] == game_id]
        fill = np.zeros((n_events-game_events.shape[0], n_columns)) # Fill with zeros
        fill[:, index_dummy] = 1 # Set the value of dummy variable to one
        X[i] = np.vstack((game_events.to_numpy(), fill))

    assert np.isnan(X).any() == False  
    return X

def preprocess_game_stats(game_stats=None, y_goal_diff=False, suffix=""):
    """ Preprocessing for the reference point models."""
    
    id_train = np.load('data/{0}/id/id_train.npy'.format(suffix))
    id_test = np.load('data/{0}/id/id_test.npy'.format(suffix))
    
    if game_stats is None:
        game_stats = pd.read_csv("data/csv/game_teams_stats.csv")

    home = game_stats.loc[game_stats['HoA']=='home']
    away = game_stats.loc[game_stats['HoA']=='away']  
    game_stats = home.merge(away, on='game_id', suffixes=['_home', '_away'])  

    relevant_columns = ['game_id', 'won_home', 'settled_in_home', 'goals_home', 'shots_home', 
                        'hits_home', 'pim_home', 'powerPlayOpportunities_home', 
                        'faceOffWinPercentage_home', 'giveaways_home', 'takeaways_home', 'goals_away', 
                        'shots_away', 'hits_away', 'pim_away', 'powerPlayOpportunities_away', 
                        'giveaways_away', 'takeaways_away']
    game_stats = game_stats.loc[:, relevant_columns]

    if not y_goal_diff:
        
        def result(x):
            if x[1] == 'OT' or x[1] == 'SO':
                return 'b_tie'
            elif x[0]:
                return 'a_home_win'
            else:
                return 'c_away_win'

        game_stats['result'] = game_stats.loc[:,['won_home', 'settled_in_home']].apply(result, axis=1)
        y_train = game_stats.loc[game_stats['game_id'].isin(id_train), 
                                'result'].astype('category').cat.codes
        y_test = game_stats.loc[game_stats['game_id'].isin(id_test), 
                                'result'].astype('category').cat.codes
    else:
        game_stats['goal_diff'] = game_stats[['settled_in_home', 'goals_home', 'goals_away']].apply(
                                            goal_difference, axis=1)
        y_train = game_stats.loc[game_stats['game_id'].isin(id_train), 'goal_diff'].to_numpy()
        y_test = game_stats.loc[game_stats['game_id'].isin(id_test), 'goal_diff'].to_numpy()

    features = ['shots_home', 'hits_home', 'pim_home', 'powerPlayOpportunities_home', 
                'faceOffWinPercentage_home', 'giveaways_home', 'takeaways_home', 'shots_away', 
                'hits_away', 'pim_away', 'powerPlayOpportunities_away', 
                'giveaways_away', 'takeaways_away']
    
    X_train = game_stats.loc[game_stats['game_id'].isin(id_train), features].to_numpy()
    X_test = game_stats.loc[game_stats['game_id'].isin(id_test), features].to_numpy()
    
    # Scale the features to have zero mean and unit variance.
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)

    return X_train, X_test, y_train, y_test

def make_index(n_games, ids, suffix):
    index = np.arange(n_games)
    index_train, index_test, id_train, id_test = train_test_split(index, ids, 
                                                    test_size=0.2, random_state=13)
    
    os.mkdir('data/{0}/index'.format(suffix))
    np.save('data/{0}/index/index_train.npy'.format(suffix), index_train)
    np.save('data/{0}/index/index_test.npy'.format(suffix), index_test)

    os.mkdir('data/{0}/id'.format(suffix))
    np.save('data/{0}/id/id_train.npy'.format(suffix), id_train)
    np.save('data/{0}/id/id_test.npy'.format(suffix), id_test)
    
    return index_train, index_test

def preprocess_y(games=None, plays=None, binary_y=False, y_goal_diff=False, 
                save_as_file=False, suffix=""):
    
    if plays is None:
        plays = pd.read_csv('data/csv/game_plays.csv')
    
    if games is None:
        games = pd.read_csv('data/csv/game.csv')

    games = games.loc[games['game_id'].isin(plays['game_id'].unique()), 
                    ['game_id', 'home_team_id', 'outcome', 'home_goals', 'away_goals']]
    games = games.sort_values(by='game_id')
    
    if not y_goal_diff:
        y = pd.get_dummies(games['outcome'].apply(result, binary=binary_y))
        y_columns = dict(zip(y.columns, range(len(y.columns))))
    else:
        y = games[['outcome', 'home_goals', 'away_goals']].apply(goal_difference, axis=1)
        y_columns = {'goal_diff': 0}

    y = y.to_numpy()
    
    index_train = np.load('data/{0}/index/index_train.npy'.format(suffix))
    index_test = np.load('data/{0}/index/index_test.npy'.format(suffix))

    suffix_y = '_goal_diff' if y_goal_diff else '_result'

    if save_as_file:
        save_data(y[index_train], y[index_test], y_columns, suffix, suffix_y)

    return y[index_train], y[index_test], y_columns                      
    

def preprocess_X(plays=None, games=None, relevant_events=None, include=[], 
               save_as_file=False, return_arrays=True, suffix=""):    
    """ Function for preprocessing the play-by-play data for analysis. """
    
    if save_as_file and os.path.exists('data/{0}'.format(suffix)):
        raise ValueError("Directory already exists with suffix name.")

    if save_as_file and not os.path.exists('data/{0}'.format(suffix)):
        os.mkdir('data/{0}'.format(suffix))

    if plays is None:
        plays = pd.read_csv('data/csv/game_plays.csv')

    if games is None:
        games = pd.read_csv('data/csv/game.csv')
    
    if relevant_events is None: # Use default events as features 
        relevant_events = ['Faceoff', 'Giveaway', 'Blocked Shot', 'Shot', 
                            'Hit', 'Penalty', 'Takeaway', 'Missed Shot']

    plays = plays[plays['event'].isin(relevant_events)] # Predict based on these events
    plays = plays[plays['period'] <= 3] # Only plays in regulation included
    plays = plays.loc[:, ['game_id', 'team_id_for', 'team_id_against', 'event', 'st_x', 'st_y']]

    # Filter central ice faceoffs
    plays = plays.loc[~((plays['st_x']==0) & (plays['st_y']==0) & (plays['event']=='Faceoff'))]
    plays = plays.dropna() # Remove rows with missing data
    
    if 'distance' in include:
        plays['distance'] = plays[['st_x', 'st_y']].apply(measures.distance, axis=1)
    
    if 'distance' in include and 'angle' in include:
        plays['angle'] = plays[['st_x', 'distance']].apply(measures.angle, axis=1)
    
    plays = pd.get_dummies(plays, columns=['event'])
    plays['event_dummy'] = 0

    plays.rename(columns=lambda x: x.replace(' ', '_').lower(), inplace=True)

    games = games.loc[games['game_id'].isin(plays['game_id'].unique()), ['game_id', 'home_team_id']]
    games = games.sort_values(by='game_id')
    plays = plays.merge(games, on='game_id')

    plays['home_team_for'] = plays[['team_id_for', 'home_team_id']].apply(home_team_for, axis=1)
    
    if 'zones' in include:
        plays['offensive_zone_home'] = plays[['st_x', 'home_team_for']].apply(
                                measures.offensive_zone_home, axis=1)
        plays['offensive_zone_away'] = plays[['st_x', 'home_team_for']].apply(
                                measures.offensive_zone_away, axis=1)

    predict_columns = ['game_id', 'st_x', 'st_y', 'distance', 'angle',
                        'home_team_for', 'offensive_zone_home', 'offensive_zone_away']

    plays = plays.loc[:, (plays.columns.isin(predict_columns) | plays.columns.str.contains('event'))]

    scale_columns = ['st_x', 'st_y', 'distance', 'angle']
    plays.loc[:, plays.columns.isin(scale_columns)] = StandardScaler().fit_transform(
                                                    plays.loc[:, plays.columns.isin(scale_columns)])

    X = to_3d_array(plays)

    if not os.path.exists('data/{0}/index'.format(suffix)):
        index_train, index_test = make_index(X.shape[0], X[:, 0, 0], suffix)
    else:
        index_train = np.load('data/index/index_train.npy')
        index_test = np.load('data/index/index_test.npy')

    X_train, X_test = X[index_train, :, 1:], X[index_test, :, 1:]

    X_columns = dict(zip(plays.columns[1:], range(len(plays.columns)-1)))
    
    if save_as_file:
        save_data(X_train, X_test, X_columns, suffix, X=True)
    
    if return_arrays:
        return X_train, X_test, X_columns