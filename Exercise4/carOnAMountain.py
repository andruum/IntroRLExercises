import numpy as np
import math
import matplotlib.pyplot as plt

class CoM(object):      # Car on Mountain
    

    def __init__(self,dt):
        self.m = 1
        self.h = 1
        self.L = 4
        self.g = 9.82
        self.dt = dt
        self.startX = self.L/2
        self.startVel = 0
        self.velGain = 4 
        self.direction = 1
        self.at = self.velGain*self.direction
        self.currentX = self.startX
        self.currentV = self.startVel
        self.currentY = self.getY(self.currentX)

        self.pos = [self.currentX]
        self.vel = [self.currentV]



    def step(self,at):
        self.at = at*self.velGain
        nextX = self.nextX(self.currentX,self.currentV)
        nextV = self.nextV(self.at,self.currentX,self.currentV)
        
        stNext = [nextX,nextV]
        rNext = self.currentX**3        # Reward function 
        self.currentX = nextX
        self.currentV = nextV

        return stNext, rNext

    
    def A(self,x):
        return self.m*(1+(pow(self.h,2)/pow(self.L,2))*pow(math.pi,2)*pow(math.sin(2*math.pi*x/self.L),2))

    def B(self,x):
        return self.m*(pow(self.h,2)/pow(self.L,3))*pow(math.pi,3)*math.sin((4*math.pi*x)/self.L)

    def C(self,x):
        return -(self.m*self.g*self.h)/self.L*math.pi*math.sin((2*math.pi*x)/self.L)

    def D(self,x):
        return math.sqrt(1+pow(math.pi,2)*pow(self.h,2)/pow(self.L,2)*(pow(math.sin(2*math.pi*x/self.L),2)))


    def nextX(self,xCur,vCur):
        return xCur + self.dt*vCur  

    def nextV(self,input,xCur,vCur):
        return vCur + (self.dt/self.A(xCur))*(-self.B(xCur)*pow(vCur,2)-self.C(xCur)+self.at/self.D(xCur))

    def getY(self,xCur):
        return self.h/2 *(1+math.cos((2*math.pi*xCur)/self.L))




if __name__ == '__main__':
    

    car = CoM(0.01)
    

    for i in range(1000):   
        tempX = car.currentX 
        car.currentX = car.nextX(car.currentX,car.currentV)
        car.currentV = car.nextV(car.direction,tempX,car.currentV)
      

        print(car.step(1))
    
        
        car.pos.append(car.currentX)
        car.vel.append(car.currentV)
    # print(car.pos)
    # print(car.vel)


    # time = np.arange(0,10,car.dt)
    # plt.plot(time,car.pos)
    # plt.show()

    #     plt.scatter(car.currentX,car.currentY)
    #     plt.pause(0.01)
    # plt.show()




    # fig, (ax1,ax2) = plt.subplots(2,2, tight_layout = True)

    # ax1[0].plot(time,states[1])
    # ax1[0].set_xlabel('Time [s]')
    # ax1[0].set_ylabel('Velocity')
    # ax1[0].grid(True)

    # ax1[1].plot(time,states[0])
    # ax1[1].set_xlabel("Time [s]")
    # ax1[1].set_ylabel("Position")

    # # ax2[0].plot(time,states[0])
    # # ax2[0].set_xlabel("Time [s] ")
    # # ax2[0].set_ylabel("y")

    # # ax2[1].plot(states[2])
    # # ax2[1].set_xlabel("y")
    # # ax2[1].set_ylabel("")



    # plt.show()



