import math

def calc_line(current_loc, target_loc,cur_ang):
    
    x = target_loc[0] - current_loc[0]
    y = target_loc[1] - current_loc[1]
    
    dist = (x**2+y**2)**0.5
    angle = math.atan2(y,x)*180/math.pi
    angle -= cur_ang
    
    if angle > 180:
        angle -= 360
    elif angle < -180:
        angle += 360
    
    return angle,dist