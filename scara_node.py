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
        self.current_pose = np.zeros(4)
        self.next_pose = np.zeros(4)
        self.end_pose = np.zeros(4)
        self.a = [0.4670,0.4005]
        self.points = [i for i in np.arange(-1.4, 1.4, 0.05)]
        self.radius = 0.8675
        self.cartesian_points = [(self.radius*np.cos(i), self.radius*np.sin(i), 0) for i in self.points]
        self.plotTrajectory()
                

    def plotTrajectory(self):
        print("Received")
        plt.plot(self.cartesian_points)
        plt.show()

    def callback(self,msg):
        self.current_pose = msg.data
        #rospy.loginfo(msg.data)

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

    def inverseJacobian(self, q):
        J = self.jacobian(q)
        jinv = np.linalg.pinv(J) 
        return jinv

    def publishing(self):
            for p in self.cartesian_points:
                X_current = self.fowardKinematics(self.current_pose)
                X_dest = [p[0], p[1], p[2]]

                error = np.subtract(X_dest, X_current)*0.6

                jinv = self.inverseJacobian(self.current_pose)
                T = np.dot(jinv, error)

                theta = self.inverseKinematics(X_dest)

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
                print(msg.data)
                rospy.sleep(0.2)
                    


if __name__ == '__main__':
    signal.signal(signal.SIGINT, signal_handler)
    sp = scara_planner()
    sp.publishing()
    rospy.spin()