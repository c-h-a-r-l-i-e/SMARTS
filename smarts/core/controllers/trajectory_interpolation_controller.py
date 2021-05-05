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

from smarts.core.controllers.trajectory_tracking_controller import (
    TrajectoryTrackingControllerState,
    TrajectoryTrackingController,
)

TIME_INDEX = 0
X_INDEX = 1
Y_INDEX = 2
THETA_INDEX = 3
VEL_INDEX = 4


class TrajectoryInterpolationController:
    @staticmethod
    def is_legal_trajectory(trajectory: np.ndarray):
        assert len(trajectory[TIME_INDEX]) >= 2, "Length of trajectory is less than 2!"

        assert np.isfinite(
            trajectory
        ).all(), "Has nan, positive infi or nagative infi in trajectory!"

        assert (
            np.diff(trajectory[TIME_INDEX]) > 0
        ).all(), "Time of trajectory is not strictly increasing!"

    @staticmethod
    def interpolate(ms0: np.ndarray, ms1: np.ndarray, time: float) -> np.ndarray:
        """Linear Interpolate between two vehicle motion state

        Returns:
            np.ndarray: New vehicle state between vehicle motion state ms0 and ms1
        """

        start_time = ms0[TIME_INDEX]
        end_time = ms1[TIME_INDEX]
        ratio = math.fabs((time - start_time) / (end_time - start_time))
        assert end_time >= start_time and time >= start_time

        np_motion_state = (1 - ratio) * ms0 + ratio * ms1
        return np_motion_state

    @staticmethod
    def locate_motion_state(trajectory, time) -> np.ndarray:
        end_index = 0
        for i, t in enumerate(trajectory[TIME_INDEX]):
            if t > time:
                end_index = i
                break

        assert (
            end_index > 0
        ), f"Expected relative time, {time} sec, can not be located at input with-time-trajectory"

        return trajectory[:, end_index - 1], trajectory[:, end_index]

    @classmethod
    def perform_trajectory_interpolation(
        cls,
        sim,
        agent_id,
        vehicle,
        trajectory: np.ndarray,
        controller_state,
    ):
        """Move vehicle by trajectory interpolation.

        Trajectory mentioned here has 5 dimensions, which are TIME, X, Y, THETA and VEL.
        TIME indicate

        If you want vehicle stop at a specific pose,
        trajectory[TIME_INDEX][0] should be set as numpy.inf

        Args:
            sim : reference of smarts instance
            agent_id : agent who use this controller
            vehicle : vehicle to be controlled
            trajectory (np.ndarray): trajectory with time

            controller_state : inner state of controller
        """
        assert isinstance(vehicle.chassis, BoxChassis)
        cls.is_legal_trajectory(trajectory)

        ms0, ms1 = cls.locate_motion_state(trajectory, sim.timestep_sec)

        speed = 0.0
        pose = []
        if math.isinf(ms0[TIME_INDEX]) or math.isinf(ms1[TIME_INDEX]):
            center_position = ms0[X_INDEX : Y_INDEX + 1]
            center_heading = Heading(ms0[THETA_INDEX])
            pose = Pose.from_center(center_position, center_heading)
            speed = 0.0
        else:
            ms = cls.interpolate(ms0, ms1, sim.timestep_sec)

            center_position = ms[X_INDEX : Y_INDEX + 1]
            center_heading = Heading(ms[THETA_INDEX])
            pose = Pose.from_center(center_position, center_heading)
            speed = ms[VEL_INDEX]

        vehicle.set_control(pose, speed)

