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
from smarts.core.controllers.pure_controller import (
    PureController,
    PureLaneFollowingController,
)


class SafetyPureController:
    @classmethod
    def perform_action(cls, sim, vehicle, action, dt): # TODO: change the call for this to match
        """
        Check if an action is safe, and if not modify it to be as safe as possible, then
        execute using pure physics (no pybullet).
        """
        assert isinstance(vehicle.chassis, BoxChassis)
        # Process the action inputs
        throttle, brake, steering_angle = action
        
        throttle = np.clip(throttle, 0, 1)
        brake = np.clip(brake, 0, 1)
        steering_angle = np.clip(steering_angle, -1, 1)

        a = throttle * vehicle.max_accel - brake * vehicle.max_brake
        delta = steering_angle * np.pi/4

        # TODO: modify actions for safety

        # 1. Calculate the surrounding roads
        lanes = SafetyPureController.get_lanes()

        # 2. For each road, calcualte surrounding vehicles
        sim.neighbourhood_vehicles_around_vehicle(vehicle = vehicle, radius = np.inf) # TODO: do we need np inf?

        # 3. Pass information to our safety calculator to work out a safe action, hopefully
        #    close to the original action intention.


        deltas = carsim.logic.calculate_safe_deltas(...)

        # Check if deltas empty
        if deltas empty:
            a = - vehicle.max_brake
            deltas = carsim.logic.calcaulte_safe_deltas(...)

            if deltas empty:
                # Set delta to follow lane somehow


        # Find closest delta




        action = (throttle, brake, steering_angle)
        PureController.perform_action(vehicle, action, dt)



class SafetyPureLaneFollowingController:
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
        # For now this method simply passes the lane following action the safety controller, which should ensure that
        # each action is individually safe.
        action = PureLaneFollowingController.get_action(sim, vehicle, sensor_state, dt, target_speed, lane_change)
        SafetyPureController.perform_action(vehicle, action, dt)


