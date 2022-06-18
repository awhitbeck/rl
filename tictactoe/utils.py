def rotate3x3(state,angle):
    if not angle in [0,90,180,270]:
        print('bad angle')
        return state
    if angle == 0:
        return state
    elif angle == 90:
        state[0],state[1],state[2],state[3],state[5],state[6],state[7],state[8]=state[2],state[5],state[8],state[1],state[7],state[0],state[3],state[6]
        return state
    else :
        return rotate3x3(state,angle-90)
