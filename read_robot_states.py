import numpy as np
import ur_rotation_module


class RobotState():

    def __init__(self, timestamp, joints, x, y, z, rx, ry, rz):
        self.timestamp = timestamp
        self.joints = joints
        self.x = x
        self.y = y
        self.z = z
        self.rx = rx
        self.ry = ry
        self.rz = rz
        zeros_one = np.matrix('0 0 0 1')
        R_tcp_base = ur_rotation_module.rot_vec2rot_mat([self.rx, self.ry, self.rz])
        t_base_tcp = np.matrix([[self.x],[self.y],[self.z]])
        R_base_tcp = R_tcp_base.transpose()
        t_tcp_base = -1.0 * R_tcp_base.dot(t_base_tcp)
        self.T_tcp_base = np.append(R_tcp_base, t_tcp_base, axis = 1)
        self.T_tcp_base = np.append(self.T_tcp_base, zeros_one, axis = 0)

        self.T_base_tcp = np.append(R_base_tcp, t_base_tcp, axis = 1)
        self.T_base_tcp = np.append(self.T_base_tcp, zeros_one, axis = 0)



    def __str__(self):
        return ("Timestamp: {}, Joint1: {}, Joint2: {}, Joint3: {}, Joint4: {}, Joint5: {}, Joint6: {}, " + 
             "x: {}, y: {}, z: {}, rx: {}, ry: {}, rz: {},").format(self.timestamp, self.joints[0],
             self.joints[1], self.joints[2], self.joints[3], self.joints[4], self.joints[5], 
             self.x, self.y, self.z, self.rx, self.ry, self.rz)

    def getR_tcp_base(self):
        return ur_rotation_module.rot_vec2rot_mat(self.rx, self.ry, self.rz)


    def getT_tcp_base(self):
        R_tcp_base = self.getR_tcp_base()
        t_base_tcp = self

    def get_rotation_vector(self):
        return [self.rx, self.ry, self.rz]




# Read robot states file


def read_robot_states(filename):
    file = open(filename, "r") 
    lines = file.readlines()

    robot_states = []
    
    #x_prev = 0.0
    #t_prev = 0.0
    for i in range(0,len(lines),13):
        if (lines[i].split(':')[0].strip()) != "Timestamp":
            raise NameError('Robot states file structure error!')
        timestamp = float(lines[i].split(':')[1].strip()) * 1000.0
        joints = []
        for j in range(6):
            joints.append(float(lines[i+1+j].split(':')[1].strip())) 
        x = float(lines[i+7].split(':')[2].strip())
        y = float(lines[i+8].split(':')[2].strip())
        z = float(lines[i+9].split(':')[2].strip())
        rx = float(lines[i+10].split(':')[2].strip())
        ry = float(lines[i+11].split(':')[2].strip())
        rz = float(lines[i+12].split(':')[2].strip())
        robot_states.append(RobotState(timestamp, joints, x, y, z, rx, ry, rz))
        
        #print((x - x_prev) / (timestamp - t_prev))
        #x_prev = x
        #t_prev = timestamp
    return robot_states

def get_velocity(robot_state, robot_state_prev):
    robot_dt = (robot_state.timestamp - \
                    robot_state_prev.timestamp) / (30.0) #WHY?

    velocity_robot = [(robot_state.x - robot_state_prev.x) / robot_dt,
                      (robot_state.y - robot_state_prev.y) / robot_dt,
                      (robot_state.z - robot_state_prev.z) / robot_dt]

    return velocity_robot

def get_dt(robot_state, robot_state_prev):
    robot_dt = (robot_state.timestamp - \
                    robot_state_prev.timestamp) / (30.0) #WHY?
    return robot_dt
    
#robot_states = read_robot_states("robot_states.txt") 
#for state in robot_states:
   # print(state)
   #print(state.x)
 

