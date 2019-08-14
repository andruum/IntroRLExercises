import numpy as np
import math
import matplotlib.pyplot as plt

m = 1
h = 1
L = 4           # L
g = 9.82        # Gravity
dt = 0.01


startX = L/2
startVel = 0
u_t = 4
input = u_t*-1


# Storage
pos = []
vel = []
y = []
time = np.arange(0,20,dt)
states = [pos,vel,y]






def A(x):
    return m*(1+(pow(h,2)/pow(L,2))*pow(math.pi,2)*pow(math.sin(2*math.pi*x/L),2))

def B(x):
    return m*(pow(h,2)/pow(L,3))*pow(math.pi,3)*math.sin((4*math.pi*x)/L)

def C(x):
    return -(m*g*h)/L*math.pi*math.sin((2*math.pi*x)/L)

def D(x):
    return math.sqrt(1+pow(math.pi,2)*pow(h,2)/pow(L,2)*(pow(math.sin(2*math.pi*x/L),2)))


def nextX(xCur,vCur):
    return xCur + dt*vCur

def nextV(input,xCur,vCur):
    return vCur + (dt/A(xCur))*(-B(xCur)*pow(vCur,2)-C(xCur)+input/D(xCur))

def getY(xCur):
    return h/2 *(1+math.cos((2*math.pi*xCur)/L))



for i in range(len(time)-1):
    if(i == 0):
        states[0].append(startX)                        # Sets start position
        states[1].append(nextV(input,states[0][i],0))   # Sets start velocity 
        states[2].append(getY(startX))                  # Sets start y-value
        #time.append(0)
    
    states[0].append(nextX(states[0][i],states[1][i]))  #
    states[1].append(nextV(input,states[0][i],states[1][i]))
    states[2].append(getY(states[0][i]))


if nextX(states[0][i],states[1][i]) > L:
    print("Goal reaced")
    

fig, (ax1,ax2) = plt.subplots(2,2, tight_layout = True)

ax1[0].plot(time,states[1])
ax1[0].set_xlabel('Time [s]')
ax1[0].set_ylabel('Velocity')
ax1[0].grid(True)

ax1[1].plot(time,states[0])
ax1[1].set_xlabel("Time [s]")
ax1[1].set_ylabel("Position")

# ax2[0].plot(time,states[0])
# ax2[0].set_xlabel("Time [s] ")
# ax2[0].set_ylabel("y")

# ax2[1].plot(states[2])
# ax2[1].set_xlabel("y")
# ax2[1].set_ylabel("")



plt.show()



