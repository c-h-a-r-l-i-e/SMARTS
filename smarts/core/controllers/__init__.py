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
from enum import Enum
from functools import partial
import ray

import numpy as np

from pathos.helpers import mp

from smarts.core.controllers.actuator_dynamic_controller import (
    ActuatorDynamicController,
    ActuatorDynamicControllerState,
)
from smarts.core.controllers.lane_following_controller import (
    LaneFollowingController,
    LaneFollowingControllerState,
)
from smarts.core.controllers.trajectory_tracking_controller import (
    TrajectoryTrackingController,
    TrajectoryTrackingControllerState,
)
from smarts.core.controllers.trajectory_interpolation_controller import (
    TrajectoryInterpolationController,
)
from smarts.core.controllers.pure_controller import (
    PureController,
    PureLaneFollowingController,
)
from smarts.core.controllers.safety_controller import (
    SafetyPureController,
    SafetyPureLaneFollowingController,
    get_safe_action,
)


METER_PER_SECOND_TO_KM_PER_HR = 3.6


class ActionSpaceType(Enum):
    Continuous = 0
    Lane = 1
    ActuatorDynamic = 2
    LaneWithContinuousSpeed = 3
    TargetPose = 4
    Trajectory = 5
    MultiTargetPose = 6  # for boid control
    MPC = 7
    TrajectoryWithTime = 8  # for pure interpolation controller
    PureContinuous = 9 # Use integration of the acceleration to control the car (no pybullet)
    PureLane = 10 # Use integration, and discreate actions
    SafetyPureContinuous = 11 # Like Pure Continuous, but change actions to be safer
    SafetyPureLane = 12 # Like Pure Lane, but change actions to be safer


class Controllers:
    @staticmethod
    def perform_safe_actions(
        sim,
        vehicles,
        actions,
        sensor_states,
        action_spaces,
    ):
        """
        Seperate method needed for safe actions, as we must collect the safe actions first,
        before later applying them, as otherwise state appears out of sync.
        """
        args = []
        for i in range(len(vehicles)):
            vehicle = vehicles[i]
            action = actions[i]
            sensor_state = sensor_states[i]
            action_space = action_spaces[i]
            other_veh_states = sim.neighborhood_vehicles_around_vehicle(vehicle=vehicle)

            if action_space == ActionSpaceType.SafetyPureLane:
                get_lane_following = partial(
                    PureLaneFollowingController.get_action,
                    sim=sim,
                    vehicle=vehicle,
                    sensor_state=sensor_state,
                    dt = sim.timestep_sec,
                )

                # 12.5 m/s (45 km/h) is used as the nominal speed for lane change.
                # For keep_lane, the nominal speed is set to 15 m/s (54 km/h).
                if action == "keep_lane":
                    action = get_lane_following(target_speed=15, lane_change=0)
                elif action == "slow_down":
                    action = get_lane_following(target_speed=0, lane_change=0)
                elif action == "change_lane_left":
                    action = get_lane_following(target_speed=12.5, lane_change=1)
                elif action == "change_lane_right":
                    action = get_lane_following(target_speed=12.5, lane_change=-1)


            if action_space in [ActionSpaceType.SafetyPureContinuous, ActionSpaceType.SafetyPureLane]:
                #safe_action = SafetyPureController.get_action(
                #    other_veh_states, sensor_state, vehicle, action, sim.timestep_sec
                #)
                #args.append( (other_veh_states, sensor_state, vehicle, action, sim.timestep_sec) )
                args.append( SafetyPureController.get_reqs(other_veh_states, sensor_state, vehicle, action, sim.timestep_sec)) 

            else:
                raise ValueError(
                    f"perform_safe_actions(action_space={action_space}, ...) has failed "
                    "inside controller"
                )

        remote = False
        if remote:
            result_ids = [get_safe_action.options(placement_group=False).remote(arg) for arg in args]
            safe_actions = ray.get(result_ids)
        else:
            safe_actions = [SafetyPureController.get_safe_action(arg) for arg in args]



        # Perform action across multiple cores
        #with mp.Pool(processes=4) as pool:
        #    safe_actions = pool.map(SafetyPureController.get_safe_action, args)

        for i in range(len(vehicles)):
            vehicle = vehicles[i]
            action = safe_actions[i]
            PureController.perform_action(vehicle, action, dt=sim.timestep_sec)



    @staticmethod
    def perform_action(
        sim,
        agent_id,
        vehicle,
        action,
        controller_state,
        sensor_state,
        action_space,
        vehicle_type,
    ):
        if action is None:
            return
        if vehicle_type == "bus":
            assert action_space == ActionSpaceType.Trajectory
        if action_space == ActionSpaceType.Continuous:
            vehicle.control(
                throttle=np.clip(action[0], 0.0, 1.0),
                brake=np.clip(action[1], 0.0, 1.0),
                steering=np.clip(action[2], -1, 1),
            )
        elif action_space == ActionSpaceType.ActuatorDynamic:
            ActuatorDynamicController.perform_action(
                vehicle, action, controller_state, dt_sec=sim.timestep_sec
            )
        elif action_space == ActionSpaceType.Trajectory:
            TrajectoryTrackingController.perform_trajectory_tracking_PD(
                action,
                vehicle,
                controller_state,
                dt_sec=sim.timestep_sec,
            )
        elif action_space == ActionSpaceType.MPC:
            TrajectoryTrackingController.perform_trajectory_tracking_MPC(
                action, vehicle, controller_state, sim.timestep_sec
            )
        elif action_space == ActionSpaceType.LaneWithContinuousSpeed:
            LaneFollowingController.perform_lane_following(
                sim,
                agent_id,
                vehicle,
                controller_state,
                sensor_state,
                action[0],
                action[1],
            )
        elif action_space == ActionSpaceType.Lane:
            perform_lane_following = partial(
                LaneFollowingController.perform_lane_following,
                sim=sim,
                agent_id=agent_id,
                vehicle=vehicle,
                controller_state=controller_state,
                sensor_state=sensor_state,
            )

            # 12.5 m/s (45 km/h) is used as the nominal speed for lane change.
            # For keep_lane, the nominal speed is set to 15 m/s (54 km/h).
            if action == "keep_lane":
                perform_lane_following(target_speed=15, lane_change=0)
            elif action == "slow_down":
                perform_lane_following(target_speed=0, lane_change=0)
            elif action == "change_lane_left":
                perform_lane_following(target_speed=12.5, lane_change=1)
            elif action == "change_lane_right":
                perform_lane_following(target_speed=12.5, lane_change=-1)
        elif action_space == ActionSpaceType.TrajectoryWithTime:
            TrajectoryInterpolationController.perform_trajectory_interpolation(
                sim, agent_id, vehicle, action, controller_state
            )
        elif action_space == ActionSpaceType.PureContinuous:
            PureController.perform_action(
                vehicle, action, dt=sim.timestep_sec
            )
        elif action_space == ActionSpaceType.PureLane:
            perform_lane_following = partial(
                PureLaneFollowingController.perform_lane_following,
                sim=sim,
                agent_id=agent_id,
                vehicle=vehicle,
                sensor_state=sensor_state,
                dt = sim.timestep_sec,
            )

            # 12.5 m/s (45 km/h) is used as the nominal speed for lane change.
            # For keep_lane, the nominal speed is set to 15 m/s (54 km/h).
            if action == "keep_lane":
                perform_lane_following(target_speed=15, lane_change=0)
            elif action == "slow_down":
                perform_lane_following(target_speed=0, lane_change=0)
            elif action == "change_lane_left":
                perform_lane_following(target_speed=12.5, lane_change=1)
            elif action == "change_lane_right":
                perform_lane_following(target_speed=12.5, lane_change=-1)
        elif action_space == ActionSpaceType.SafetyPureContinuous:
            SafetyPureController.perform_action(
                sim, sensor_state, vehicle, action, dt=sim.timestep_sec
            )
        elif action_space == ActionSpaceType.SafetyPureLane:
            perform_lane_following = partial(
                SafetyPureLaneFollowingController.perform_lane_following,
                sim=sim,
                agent_id=agent_id,
                vehicle=vehicle,
                sensor_state=sensor_state,
                dt = sim.timestep_sec,
            )

            # 12.5 m/s (45 km/h) is used as the nominal speed for lane change.
            # For keep_lane, the nominal speed is set to 15 m/s (54 km/h).
            if action == "keep_lane":
                perform_lane_following(target_speed=15, lane_change=0)
            elif action == "slow_down":
                perform_lane_following(target_speed=0, lane_change=0)
            elif action == "change_lane_left":
                perform_lane_following(target_speed=12.5, lane_change=1)
            elif action == "change_lane_right":
                perform_lane_following(target_speed=12.5, lane_change=-1)
        else:
            raise ValueError(
                f"perform_action(action_space={action_space}, ...) has failed "
                "inside controller"
            )

class ControllerState:
    @staticmethod
    def from_action_space(action_space, vehicle_pose, sim):
        if action_space == ActionSpaceType.Lane:
            # TAI: we should probably be fetching these waypoint through the mission planner
            target_lane_id = sim.waypoints.closest_waypoint(
                vehicle_pose, filter_from_count=4
            ).lane_id
            return LaneFollowingControllerState(target_lane_id)

        if action_space == ActionSpaceType.LaneWithContinuousSpeed:
            # TAI: we should probably be fetching these waypoint through the mission planner
            target_lane_id = sim.waypoints.closest_waypoint(
                vehicle_pose, filter_from_count=4
            ).lane_id
            return LaneFollowingControllerState(target_lane_id)

        if action_space == ActionSpaceType.ActuatorDynamic:
            return ActuatorDynamicControllerState()

        if action_space == ActionSpaceType.Trajectory:
            return TrajectoryTrackingControllerState()

        if action_space == ActionSpaceType.MPC:
            return TrajectoryTrackingControllerState()

        # Other action spaces do not need a controller state object
        return None
