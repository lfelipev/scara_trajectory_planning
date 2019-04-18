#!/usr/bin/env python
import rospy
from std_msgs.msg import Float64MultiArray, Float64
import numpy as np
import signal
import sys
import matplotlib.pyplot as plt

def signal_handler(sig, frame):
    print('Simulation closed')
    sys.exit(0)


class scara_planner:
    def __init__(self):
        self.pub = rospy.Publisher('ros_scara', Float64MultiArray, queue_size=10)   
        rospy.Subscriber('scara_ros', Float64MultiArray, self.callback)
        rospy.init_node('scara_joints', anonymous=True)
        # Force control 
        rospy.Subscriber('MTB_Inertia',Float64MultiArray,self.setInertia)
        rospy.Subscriber('MTB_Pos', Float64MultiArray, self.callback)
        rospy.Subscriber('MTB_Pos', Float64MultiArray, self.setVelocity)
        self.current_pose = np.zeros(4)
        self.next_pose = np.zeros(4)
        self.end_pose = np.zeros(4)
        self.a = [0.4670,0.4005]
        self.velocity = np.zeros(4)
        self.mass = [14.274293899536, 7.1078543663025,3.9662611484528 ,0.055354863405228]
        self.Iz = np.zeros(4)
        self.points = [i for i in np.arange(-1.8, 1.8, 0.05)]
        self.radius = 0.45
        self.g = np.array([0,0,0,self.mass[3]]).T
        self.cartesian_points = [(self.radius*np.cos(i), self.radius*np.sin(i), 0) for i in self.points]
        self.plotTrajectory()
        
                

    def plotTrajectory(self):
        print("Received")
        plt.plot(self.cartesian_points)
        plt.show()

    def callback(self,msg):
        self.current_pose = msg.data
        #rospy.loginfo(msg.data)
    def setInertia(self,msg):
        self.Iz = msg.data
        rospy.loginfo(msg.data)
    
    def setVelocity(self,msg):
        self.velocity = msg.data

    def fowardKinematics(self, q):
        pos = np.zeros(3)
        pos[0] = self.a[0]*np.cos(q[0]) + self.a[1]*np.cos(q[0] + q[1])
        pos[1] = self.a[0]*np.sin(q[0]) + self.a[1]*np.sin(q[0] + q[1])
        pos[2] = q[2]
        return pos

    def inverseKinematics(self, p):
        theta = np.zeros(4)
        px = p[0]
        py = p[1]
        pz = p[2]
        a1 = self.a[0]
        a2 = self.a[1]

        # Second joint
        c2 = (pow(px, 2) + pow(py, 2) - pow(a1, 2) - pow(a2, 2)) / (2*a1*a2)
        
        if (c2 >= -1) and (c2 <= 1):
            s2 = np.sqrt(1 - pow(c2, 2))
        else:
            s2 = 0 
        
        theta2 = np.arctan2(s2, c2)

        # First joint
        s1 = ((a1 + a2*c2)*py - a2*s2*px) / (pow(px, 2) + pow(py, 2))

        c1 = ((a1 + a2*c2)*px + a2*s2*py) / (pow(px, 2) + pow(py, 2))

        theta1 = np.arctan2(s1, c1)

        # Third joint
        theta3 = pz 

        # End effector
        theta4 = 0

        theta = [theta1, theta2, theta3, theta4]

        return theta
        
    def velocities(self, q):
         return q
    
    def degree(self, x):
         return x*180/np.pi
         
    def jacobian(self, q):
        return [[-self.a[0]*np.sin(q[0])-self.a[1]*np.sin(q[0] + q[1]),   -self.a[1]*np.sin(q[0] + q[1]), 0, 0],
                [ self.a[0]*np.cos(q[0])+self.a[1]*np.cos(q[0] + q[1]),    self.a[0]*np.cos(q[0] + q[1]), 0, 0],
                [0, 0, 1, 0]]

    def dot_jacobian(self,q,dot_q):
        return [[-self.a[0]*np.cos(q[0])*dot_q[0]*-self.a[1]*np.cos(q[0] + q[1])*(dot_q[0]+dot_q[1]),   -self.a[1]*np.cos(q[0] + q[1])*(dot_q[0]+dot_q[1]), 0, 0],
                [ -self.a[0]*np.sin(q[0])*dot_q[0]-self.a[1]*np.sin(q[0] + q[1])*(dot_q[0]+dot_q[1]),    -self.a[0]*np.sin(q[0] + q[1])*(dot_q[0]+dot_q[1]), 0, 0],
                [0, 0, 0, 0]]

    def inverseJacobian(self, q):
        J = self.jacobian(q)
        jinv = np.linalg.pinv(J) 
        return jinv

    def coriolis(self,q):
        dot_q = self.velocity
        alfa = self.Iz[0]+(self.a[0]/2.0)**2*self.mass[0]+self.a[0]**2*self.mass[1]+self.a[0]**2*self.mass[2]+self.a[0]**2*self.mass[3]
        beta = self.Iz[1]+self.Iz[2]+self.Iz[3]+self.a[1]**2*self.mass[2]+self.a[1]**2*self.mass[3]+self.mass[1]*(self.a[1]/2.0)**2
        lambdaa = self.a[0]*self.a[1]*self.mass[2]+self.a[0]*self.a[1]*self.mass[3]+self.a[0]*self.mass[1]*(self.a[1]/2.0)
        gama = self.Iz[2]+self.Iz[3]

        
        return [[-lambdaa*np.sin(q[1]*dot_q[1]),-lambdaa*np.sin(q[1])*(dot_q[0]+dot_q[1]),0,0],
                [lambdaa*np.sin(q[1]*dot_q[0]),0,0,0],
                [0,0,0,0],
                [0,0,0,0]]

    def inertia(self,q):
        alfa = self.Iz[0]+(self.a[0]/2.0)**2*self.mass[0]+self.a[0]**2*self.mass[1]+self.a[0]**2*self.mass[2]+self.a[0]**2*self.mass[3]
        beta = self.Iz[1]+self.Iz[2]+self.Iz[3]+self.a[1]**2*self.mass[2]+self.a[1]**2*self.mass[3]+self.mass[1]*(self.a[1]/2.0)**2
        lambdaa = self.a[0]*self.a[1]*self.mass[2]+self.a[0]*self.a[1]*self.mass[3]+self.a[0]*self.mass[1]*(self.a[1]/2.0)
        gama = self.Iz[2]+self.Iz[3]

        return [[lambdaa+beta+2*lambdaa*np.cos(q[1]),beta+lambdaa*np.cos(q[1]),gama,0],
                [beta+lambdaa*np.cos(q[1]),beta,gama,0],
                [lambdaa,lambdaa,lambdaa,0],
                [0,0,0,self.mass[3]]]
        



    def publishing(self):
            for p in self.cartesian_points:
                X_current = self.fowardKinematics(self.current_pose)
                X_dest = [p[0], p[1], p[2]]

                error = np.subtract(X_dest, X_current)*0.6

                jinv = self.inverseJacobian(self.current_pose)
                T = np.dot(jinv, error)

                theta = self.inverseKinematics(X_dest)

                Ja = self.jacobian(self.current_pose)
                X_dot = Ja * self.velocity
                
                Ja_t = np.transpose(Ja)

                # print(np.dot(Ja_t, X_dot))
                print(np.dot(Ja_t, error))
                
                Kp = np.eye(3)
                Kd = np.eye(4)



                #Bolo = np.dot(Mid - Top, Ja_t)

                # u = self.g + np.dot(Ja_t, (-np.dot(X_dot, Kd) + np.dot(error, Kp)))
                # X_jacob = np.subtract(X_dot, error)
                # X_pos_jacob = Ja_t * X_jacob

                # u = X_pos_jacob + self.g
                #print(u)

                msg = Float64MultiArray()
                if(np.abs(T[0]) >= 0.4 or np.abs(T[1] >= 0.4)):
                    if(T[0] > 1):
                        T[0] = 0.02
                    else:
                        T[0] = -0.02
                    
                    if(T[1] > 1):
                        T[1] = 0.02
                    else:
                        T[1] = -0.02
                
                msg.data = [T[0], T[1], T[2], T[3],
                            theta[0], theta[1], theta[2], theta[3]]

                
                self.pub.publish(msg)
                #print(msg.data)
                rospy.sleep(0.2)
                    


if __name__ == '__main__':
    signal.signal(signal.SIGINT, signal_handler)
    sp = scara_planner()
    sp.publishing()
    rospy.spin()
