# Please only modify the indicated area below!

from math import *
import random
import cv2
import numpy as np

landmarks = [[200.0, 200.0], [800.0, 800.0], [200.0, 800.0], [800.0, 200.0],
             [250.0, 250.0], [300.0, 300.0],
             [750.0, 750.0], [700.0, 700.0]]
world_size = 1000.0

image_world = np.zeros((1000, 1000, 3), np.uint8)
for i in range(0, len(landmarks)):
    my_tuple_loctation = (int(landmarks[i][0]), int(landmarks[i][1]))
    image_world = cv2.circle(image_world, my_tuple_loctation, 20, (0, 0, 255),-1)
cv2.namedWindow('image_world', cv2.WINDOW_NORMAL)
cv2.resizeWindow('image_world', 1000, 1000)
cv2.imshow('image_world', image_world)
cv2.waitKey(1)
writer = cv2.VideoWriter('ParticleFilterLocalization.mp4', cv2.VideoWriter_fourcc(*'MP4V'), 25, (1000, 1000), True)


class robot:
    def __init__(self):
        self.x = random.random() * world_size
        self.y = random.random() * world_size
        self.orientation = random.random() * 2.0 * pi
        self.forward_noise = 0.0
        self.turn_noise = 0.0
        self.sense_noise = 0.0

    def set(self, new_x, new_y, new_orientation):
        if new_x < 0 or new_x >= world_size:
            raise ValueError, 'X coordinate out of bound'
        if new_y < 0 or new_y >= world_size:
            raise ValueError, 'Y coordinate out of bound'
        if new_orientation < 0 or new_orientation >= 2 * pi:
            raise ValueError, 'Orientation must be in [0..2pi]'
        self.x = float(new_x)
        self.y = float(new_y)
        self.orientation = float(new_orientation)

    def set_noise(self, new_f_noise, new_t_noise, new_s_noise):
        # makes it possible to change the noise parameters
        # this is often useful in particle filters
        self.forward_noise = float(new_f_noise);
        self.turn_noise = float(new_t_noise);
        self.sense_noise = float(new_s_noise);

    def sense(self):
        Z = []
        for i in range(len(landmarks)):
            dist = sqrt((self.x - landmarks[i][0]) ** 2 + (self.y - landmarks[i][1]) ** 2)
            dist += random.gauss(0.0, self.sense_noise)
            Z.append(dist)
        return Z

    def move(self, turn, forward):
        if forward < 0:
            raise ValueError, 'Robot cant move backwards'

            # turn, and add randomness to the turning command
        orientation = self.orientation + float(turn) + random.gauss(0.0, self.turn_noise)
        orientation %= 2 * pi

        # move, and add randomness to the motion command
        dist = float(forward) + random.gauss(0.0, self.forward_noise)
        x = self.x + (cos(orientation) * dist)
        y = self.y + (sin(orientation) * dist)
        x %= world_size  # cyclic truncate
        y %= world_size

        # set particle
        res = robot()
        res.set(x, y, orientation)
        res.set_noise(self.forward_noise, self.turn_noise, self.sense_noise)
        return res

    def Gaussian(self, mu, sigma, x):

        # calculates the probability of x for 1-dim Gaussian with mean mu and var. sigma
        return exp(- ((mu - x) ** 2) / (sigma ** 2) / 2.0) / sqrt(2.0 * pi * (sigma ** 2))

    def measurement_prob(self, measurement):

        # calculates how likely a measurement should be

        prob = 1.0;
        for i in range(len(landmarks)):
            dist = sqrt((self.x - landmarks[i][0]) ** 2 + (self.y - landmarks[i][1]) ** 2)
            prob *= self.Gaussian(dist, self.sense_noise, measurement[i])
        return prob

    def __repr__(self):
        return '[x=%.6s y=%.6s orient=%.6s]' % (str(self.x), str(self.y), str(self.orientation))


def eval(r, p):
    sum = 0.0;
    for i in range(len(p)):  # calculate mean error
        dx = (p[i].x - r.x + (world_size / 2.0)) % world_size - (world_size / 2.0)
        dy = (p[i].y - r.y + (world_size / 2.0)) % world_size - (world_size / 2.0)
        err = sqrt(dx * dx + dy * dy)
        sum += err
    return sum / float(len(p))

myrobot = robot()

rand_orientation = random.random() * 2*pi
rand_motion = random.random() * 5
myrobot = myrobot.move(rand_orientation,rand_motion)
Z = myrobot.sense()
N = 1000
T = 3000  # Leave this as 10 for grading purposes.

p = []
for i in range(N):
    r = robot()
    r.set_noise(.8, .8, 5.0)
    p.append(r)

error = eval(myrobot, p)
print(error)

c = 0


for t in range(T):
    rand_orientation = random.random()* 0 + 0.01
    rand_motion = random.random() * 15
    myrobot = myrobot.move(rand_orientation,rand_motion)
    canvas_to_draw = np.copy(image_world)

    Z = myrobot.sense()

    p2 = []
    for i in range(N):
        p2.append(p[i].move(rand_orientation,rand_motion))
    p = p2

    for i in range(len(p)):
        my_tuple_loctation = (int(p[i].y), int(p[i].x))
        canvas_to_draw = cv2.circle(canvas_to_draw, my_tuple_loctation, 20, (255, 255, 255), -1)

    my_tuple_loctation = (int(myrobot.y), int(myrobot.x))
    canvas_to_draw = cv2.circle(canvas_to_draw, my_tuple_loctation, 20, (0, 255, 0), -1)

    for i in range(0, len(landmarks)):
        my_robot_loctation = (int(myrobot.y), int(myrobot.x))
        my_landmark_loctation = (int(landmarks[i][0]), int(landmarks[i][1]))
        canvas_to_draw = cv2.line(canvas_to_draw, my_robot_loctation, my_landmark_loctation, (255, 0, 0), 4)

    if c == 0:
        cv2.imshow('image_world', canvas_to_draw)
        cv2.waitKey(200)
        c += 1
    else :
        writer.write(canvas_to_draw)
        cv2.imshow('image_world', canvas_to_draw)
        cv2.waitKey(20)


    w = []
    for i in range(N):
        w.append(p[i].measurement_prob(Z))

    p3 = []
    index = int(random.random() * N)
    beta = 0.0
    mw = max(w)
    for i in range(N):
        beta += random.random() * 2.0 * mw
        while beta > w[index]:
            beta -= w[index]
            index = (index + 1) % N
        p3.append(p[index])
    p = p3

    error = eval(myrobot, p)



