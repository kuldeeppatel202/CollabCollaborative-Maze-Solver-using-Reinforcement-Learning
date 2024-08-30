from controller import Robot, Compass
import math , time
import mazemodel
import dist_angle
import random

robot = Robot()

timestep = 32

maxvel = 6.28

left_wheel = robot.getDevice("left wheel motor")
right_wheel = robot.getDevice("right wheel motor")

left_wheel.setPosition(float("inf"))
right_wheel.setPosition(float("inf"))

left_wheel.setVelocity(0.0)
right_wheel.setVelocity(0.0)

compass = robot.getDevice("compass")
compass.enable(timestep)

gps = robot.getDevice('gps')
gps.enable(timestep)

E_puck_emitter = robot.getDevice('emitter')
E_puck_receiver = robot.getDevice('receiver')

E_puck_receiver.enable(timestep)

ps_names = ["ps0", "ps1", "ps2", "ps3", "ps4", "ps5", "ps6", "ps7"]
ps = [robot.getDevice(name) for name in ps_names]
for sensor in ps:
    sensor.enable(timestep)

model = mazemodel.maze()

def go_straight(distance, velocity = 2, radius = 0.0205):
    
    global timestep
    
    t = distance / (radius * velocity)
    
    start_time = robot.getTime()
    while robot.step(timestep) != -1:
        current_time = robot.getTime()

        elapsed_time = current_time - start_time
    
        if elapsed_time >= t:
            left_wheel.setVelocity(0)
            right_wheel.setVelocity(0) 
            break
        
        left_wheel.setVelocity(velocity)
        right_wheel.setVelocity(velocity)        
    
    
def turn(start_angle, target_angle=90,vel=3.14 ):
    global compass, robot, left_wheel, right_wheel,timestep
    if target_angle < 0 :
        flag = 1  # Clockwise
        target_angle = abs(target_angle)
    elif target_angle >= 0:
        flag = 0 # Anti-clockwise
        
    first = 1
    neg = 0
    pos = 0        
    while robot.step(timestep) != -1:
        orientation = compass.getValues()
        
        current_angle = (180 / math.pi) * math.atan2(orientation[0], orientation[1])
        
        if current_angle < 0 and first:
            neg = 1
            first = 0
        elif current_angle > 0 and first:
            pos = 1
            first = 0

        if flag and neg and current_angle > 0:
            current_angle -= 360 
        if not flag and pos and current_angle < 0:
            current_angle += 360
               
        if abs(current_angle-start_angle) >= target_angle:
            # Stop the wheels
            left_wheel.setVelocity(0.0)
            right_wheel.setVelocity(0.0)
            break
        
        # Rotate Anti-clockwise
        if flag == 0:
            left_wheel.setVelocity(-vel)
            right_wheel.setVelocity(vel)
        
        # Rotate Cloclwise
        elif flag == 1:
            left_wheel.setVelocity(vel)
            right_wheel.setVelocity(-vel) 

while robot.step(timestep) != -1:
   
    coord = gps.getValues()
   
    orientation = compass.getValues()
    current_angle = (180 / math.pi) * math.atan2(orientation[0], orientation[1])
    
    ps_values = [sensor.getValue() for sensor in ps]

    tup,cell_num = model.find_cell(coord)
    
    message = ps_values + coord + [int(cell_num)]
    
    
   # print("RRR----", message)
    E_puck_emitter.send(message)
    # Feed the output 'cell_num' to DQN
    # DQN gives output direction which is stored in variable 'dir'
    robot.step(32)
    
    #while E_puck_receiver.getQueueLength()==0:
        #time.sleep(0.032)
    if E_puck_receiver.getQueueLength()>0:
        recevied_message = E_puck_receiver.getInts()
        E_puck_receiver.nextPacket()
        action = recevied_message[0]
        
    else:
        #print("no action recieved")
        continue    
    dict={
        0 : 'East',
        1 : 'West',
        2 : 'North',
        3 : 'South',
    }    
    dir = action
    

    #Calculate target cell based on direction  given by DQN
    target_cell = model.calc_target(dir, cell_num)
    print(f"robot_2 is going to {target_cell}")
          
    # Center of the target cell
    target_coord = model.find_center(target_cell)
    
    # Calculate distance and angle to reach the target cell from current location
    angle, dist = dist_angle.calc_line(coord, target_coord, current_angle)
        
    turn(current_angle, angle,0.25)
    go_straight(dist,0.25)
    


