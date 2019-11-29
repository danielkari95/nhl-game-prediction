import numpy as np

def distance(x):
    return np.sqrt((x[0]-90.0)**2+x[1]**2)

def angle(x):
    if 90.0-x[0] == x[1]:
        return 0
    else:
        return np.degrees(np.arccos((90.0-x[0])/x[1]))
        
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