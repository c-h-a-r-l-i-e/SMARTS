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


class PureLaneFollowingController:
    @classmethod
    def perform_lane_following(
        cls,
        sim,
        agent_id,
        vehicle,
        controller_state,
        sensor_state,
        dt,
        target_speed=12.5,
        lane_change=0,
    ):

        wp_paths = sensor_state.mission_planner.waypoint_paths_at(
            sim, vehicle.pose, lookahead=16
        )

        current_lane = PureLaneFollowingController.find_current_lane(
            wp_paths, vehicle.position
        )

        wp_path = wp_paths[np.clip(current_lane + lane_change, 0, len(wp_paths) - 1)]

        # FOR NOW: SIMPLE CALCULATION, change turning angle to follow next waypoints heading.


        lookahead = 4
        wp_headings = [wp.heading for wp in wp_path[0 : min(lookahead, len(wp_path))]]
        heading_diff =  np.average(wp_headings) - vehicle.pose.heading

        steering_angle = np.clip(heading_diff / (np.pi/4), -1, 1)

        # TODO: this is temporary speed control
        throttle = brake = 0

        action = (throttle, brake, steering_angle)


        PureController.perform_action(vehicle, action, dt)

    @staticmethod
    def find_current_lane(wp_paths, vehicle_position):
        relative_distant_lane = [
            np.linalg.norm(wp_paths[idx][0].pos - vehicle_position[0:2])
            for idx in range(len(wp_paths))
        ]
        return np.argmin(relative_distant_lane)













