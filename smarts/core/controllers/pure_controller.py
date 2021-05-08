# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
import math
import numpy as np

from smarts.core.coordinates import Heading, Pose
from smarts.core.chassis import BoxChassis


class PureController:
    @classmethod
    def perform_action(cls, vehicle, action, dt):
        """
        Move vehicle using "pure physics", i.e. don't simulate using pybullet.

        We assume the car is a fixed object which can accelerate at some level, and 
        turn using Ackermann geometry.
        """
        assert isinstance(vehicle.chassis, BoxChassis)
        # Process the action inputs
        throttle, brake, steering_angle = action
        
        throttle = np.clip(throttle, 0, 1)
        brake = np.clip(brake, 0, 1)
        steering_angle = np.clip(steering_angle, -1, 1)

        max_brake = -4
        max_accel = -4
        a = throttle * max_accel - brake * max_brake
        delta = steering_angle * np.pi/4

        # These coordinates and heading are of the centre of the car, and theta is in radians,
        # with 0 facing along the x axis.
        x = vehicle.pose.position[0]
        y = vehicle.pose.position[1]
        theta = vehicle.pose.heading + np.pi/2
        v = vehicle.speed

        w = vehicle.width
        l = vehicle.length

        # Transform to the centre of the rear axle
        x -= l/2 * np.cos(theta)
        y -= l/2 * np.sin(theta)

        # Hopefully this term prevents the car from rolling backwards
        if v < 1/36 and a < 0:
            a = 0

        if delta == 0:
            distance = v * dt + 1/2 * a * dt ** 2
            x += distance * np.cos(theta) 
            y += distance * np.sin(theta) 

        else:
            # These equations are the integrals of the kinematic equations for a 
            # Ackermann steering geometry
            x += l / np.tan(delta) * np.sin(a * np.tan(delta) * (dt**2) / (2 * l) + \
                    np.tan(delta) * dt * v / l) * np.cos(theta) \
                + l / np.tan(delta) * np.cos(a * np.tan(delta) * (dt**2) / (2 * l) + \
                        np.tan(delta) * dt * v / l) * np.sin(theta) \
                - l / np.tan(delta) * np.sin(theta)

            y += -l / np.tan(delta) * np.cos(a * np.tan(delta) * (dt**2) / (2 * l) + \
                    np.tan(delta) * dt * v / l) * np.cos(theta) \
                + l / np.tan(delta) * np.sin(a * np.tan(delta) * (dt**2) / (2 * l) + \
                                   np.tan(delta) * dt * v / l) * np.sin(theta) \
                + l / np.tan(delta) * np.cos(theta)


        theta += a * dt**2 * np.tan(delta) / (2 * l) + dt * v * np.tan(delta) / l

        v += a * dt
        v = max(v, 0)

        # Transform back to the centre of the vehicle
        x += l/2 * np.cos(theta)
        y += l/2 * np.sin(theta)
        heading = theta - np.pi/2


        pose = Pose.from_center([x, y], Heading(heading))
        vehicle.control(pose, v)
        vehicle.steering = delta


