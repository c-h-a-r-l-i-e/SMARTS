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

import carsim.logic
import carsim.efficientlogic
import sympy as sym
from sympy import S

import time

import ray

DEBUG = False
if DEBUG:
    import matplotlib.pyplot as plt
    import carsim.plot
    fig, (ax0, ax1) = plt.subplots(2)
    ax0.set_aspect("equal")
    ax1.set_aspect("equal")

#TODO: create new safety provider which will first gather actions, and then perform them!

class CarsimCar(carsim.logic.CarFuture):
    """
    An object which carsim.logic can take in order to calcualte safe actions
    """
    def __init__(self, vehicle_state, lane_start, lane_heading, a=None):
        """
        Export a car to be used in carsim code, translating it relative to the provided lane.

        params
        ---------
        vehicle_state: VehicleState
            The vehicle we want to translate
        lane_start: np array
            The starting position of the lane
        lane_heading: float
            The heading of the lane
        a : float
            The vehicle acceleration
        """
        self.length = vehicle_state.dimensions.length
        self.width = vehicle_state.dimensions.width

        position = np.array(vehicle_state.pose.position[:2])
        position -= lane_start

        c, s = np.cos(-lane_heading), np.sin(-lane_heading)
        rot = np.array(((c, -s), (s, c)))
        position=rot.dot(position)

        self.theta = (vehicle_state.pose.heading + np.pi / 2) - lane_heading
        # translate to rear bumper
        position -= np.array((np.cos(self.theta), np.sin(self.theta))) * self.length/2

        self.x = position[0]
        self.y = position[1]
        self.v = vehicle_state.speed
        self.a = a
        self.brake_max = -vehicle_state.max_brake
        self.a_max = vehicle_state.max_accel

    def __str__(self):
        return "Veh@({:.2f}, {:.2f}) {:.2f}".format(self.x, self.y, self.theta)


class SafeDeltaRequirements:
    """
    This class contains the objects required for calculating safe actions for a vehicle
    """
    def __init__(self, ego_car, lane_bounds, surroundings, dt, start_heading, heading, delta, vehicle_id):
        self.ego_car = ego_car
        self.lane_bounds = lane_bounds
        self.surroundings = surroundings
        self.dt = dt
        self.start_heading = start_heading
        self.heading = heading
        self.delta = delta
        self.vehicle_id = vehicle_id


@ray.remote
def get_safe_action(reqs):
    return SafetyPureController.get_safe_action(reqs)


class SafetyPureController:
    @staticmethod
    def get_reqs(vehicles, sensor_state, vehicle, action, dt):
        # Process the action inputs
        start_t = time.perf_counter()

        throttle, brake, steering_angle = action
        
        throttle = np.clip(throttle, 0, 1)
        brake = np.clip(brake, 0, 1)
        steering_angle = np.clip(steering_angle, -1, 1)

        a = throttle * vehicle.max_accel - brake * vehicle.max_brake
        delta = steering_angle * np.pi/4

        road_network = sensor_state.mission_planner._road_network

        # Calculate safe actions
        # 1. Calculate ego vehicle surrounding lanes list
        front_bumper = vehicle.pose.position[:2]
        current_lane = road_network.nearest_lane(front_bumper)
        surrounding_lanes = current_lane.getEdge().getLanes()
        
        surrounding_lanes_sets = [{*lane.getIncoming(), lane} for lane in surrounding_lanes] # extend with road predecessors
        all_surrounding_lanes_set = set.union(*surrounding_lanes_sets)

        # 2. For each vehicle, work out which lanes they're in
        vehicle_lanes = {v.vehicle_id: SafetyPureController.get_vehicle_lanes(v, road_network) for v in vehicles}

        # filter out vehicles not in the surrounding lanes
        vehicles = [v for v in vehicles if not vehicle_lanes[v.vehicle_id].isdisjoint(all_surrounding_lanes_set)]

        # 3. Order vehicles based on distance along ego vehicle's road
        lane_local_offsets = {v.vehicle_id: SafetyPureController.get_distance_into_lane(current_lane, v, road_network) for v in vehicles}
        vehicles.sort(key = lambda v : lane_local_offsets[v.vehicle_id])

        ego_pos = np.clip(road_network.offset_into_lane(current_lane, vehicle.pose.position[:2]), 0, current_lane.getLength())

        # Calculate the starting point/angles for the current lane
        start_point = road_network.world_coord_from_offset(current_lane, ego_pos)
        start_vector = road_network.lane_vector_at_offset(current_lane, ego_pos)
        start_heading = np.arctan2(start_vector[1], start_vector[0])


        # 4. Generate the surroundings and convert to CarsimCar
        surroundings = [[None, None] for i in range(len(surrounding_lanes))]
        for v in vehicles:
            for lane_num, surrounding_lane_set in enumerate(surrounding_lanes_sets):
                if not vehicle_lanes[v.vehicle_id].isdisjoint(surrounding_lane_set):
                    if lane_local_offsets[v.vehicle_id] < ego_pos:
                        # Car is in surrounding lane, and behind ego vehicle
                        surroundings[lane_num][0] = CarsimCar(v, start_point, start_heading)
                    else:
                        # Car is in surrounding lane, and ahead of ego vehicle
                        if surroundings[lane_num][1] is None:
                            surroundings[lane_num][1] = CarsimCar(v, start_point, start_heading)
        ego_car = CarsimCar(vehicle.state, start_point, start_heading)


        # 5. Work out the lane boundaries
        lane_bounds = []
        bound = 0
        centre = None
        for lane in surrounding_lanes:
            width = lane.getWidth()
            lane_bounds.append((bound, bound+width))
            if lane == current_lane:
                centre = bound + width / 2
            bound += width

        if centre is None:
            raise ValueError("Current lane is not in the surrounding lanes")

        for i, lb in enumerate(lane_bounds):
            lane_bounds[i] = (lb[0] - centre, lb[1] - centre)

        start_t = time.perf_counter()
        ego_car.a = a

        reqs = SafeDeltaRequirements(ego_car, lane_bounds, surroundings, dt, start_heading, vehicle.pose.heading, delta, vehicle.id)

        return reqs

    @staticmethod
    def get_safe_action(reqs):
        ego_car = reqs.ego_car
        lane_bounds = reqs.lane_bounds
        surroundings = reqs.surroundings
        dt = reqs.dt
        heading = reqs.heading
        start_heading = reqs.start_heading
        delta = reqs.delta
        vehicle_id = reqs.vehicle_id
        a = ego_car.a

        use_efficient = True
        if use_efficient:
            # use the efficient module to get a ndarray of safe deltas
            deltas = carsim.efficientlogic.get_safe_deltas(ego_car, lane_bounds, surroundings, dt)

            if len(deltas) == 0:
                a = ego_car.brake_max
                ego_car.a = a
                deltas = carsim.efficientlogic.get_safe_deltas(ego_car, lane_bounds, surroundings, dt)
                if len(deltas) == 0:
                    # There's no safe delta while max braking, so steer towards lane centre.
                    print("no safe deltas")
                    delta =  heading - start_heading + np.pi/2

            if len(deltas != 0):
                delta = deltas[np.argmin(np.abs(deltas - delta))]
                # print("picked safe delta = {}, a = {}".format(delta, a))

            if DEBUG:
                if vehicle_id[6] == "2":
                    ax0.clear()
                    carsim.plot.plot_surroundings_and_deltas(ego_car, surroundings, lane_bounds, dt, deltas, ax0)
                if vehicle_id[6] == "3":
                    ax1.clear()
                    carsim.plot.plot_surroundings_and_deltas(ego_car, surroundings, lane_bounds, dt, deltas, ax1)
                    plt.pause(0.01)

        else:
            deltas = carsim.logic.get_safe_deltas(ego_car, lane_bounds, surroundings, dt)

            # Check if deltas empty
            if deltas == S.EmptySet:
                a = - ego_car.brake_max
                ego_car.a = a
                deltas = carsim.logic.get_safe_deltas(ego_car, lane_bounds, surroundings, dt)

                if deltas == S.EmptySet:
                    # There's no safe delta while max braking, so steer towards lane centre.
                    delta =  -start_heading + heading + np.pi/2
                
            if deltas != S.EmptySet and not deltas.contains(delta):
                # Action is deemed unsafe, so find the closest possible delta, by extracting the boundary of 
                # the delta set, and picking the closest boundary
                boundary = deltas.boundary
                new_delta = float(min(boundary.args, key=lambda d : abs(d-delta)))
                safety_thresh = 0.05
                if new_delta > delta:
                    delta = new_delta + safety_thresh
                else:
                    delta = new_delta - safety_thresh

        if a > 0:
            brake = 0
            throttle = a / ego_car.a_max
        else:
            brake = a / ego_car.brake_max
            throttle = 0

        # TODO: check braking is consistent!!!

        steering_angle = delta / (np.pi/4) 

        action = (throttle, brake, steering_angle)

        return action


    @staticmethod
    def get_action(vehicles, sensor_state, vehicle, action, dt):
        """
        Retrieve a safe action for the given vehicle. This controller atempts to ensure:
            a) the action is safe
            b) the action is as close to the original action as possible
        """
        # Process the action inputs
        start_t = time.perf_counter()

        throttle, brake, steering_angle = action
        
        throttle = np.clip(throttle, 0, 1)
        brake = np.clip(brake, 0, 1)
        steering_angle = np.clip(steering_angle, -1, 1)

        a = throttle * vehicle.max_accel - brake * vehicle.max_brake
        delta = steering_angle * np.pi/4

        road_network = sensor_state.mission_planner._road_network

        # Calculate safe actions
        # 1. Calculate ego vehicle surrounding lanes list
        front_bumper = vehicle.pose.position[:2]
        current_lane = road_network.nearest_lane(front_bumper)
        surrounding_lanes = current_lane.getEdge().getLanes()
        
        surrounding_lanes_sets = [{*lane.getIncoming(), lane} for lane in surrounding_lanes] # extend with road predecessors
        all_surrounding_lanes_set = set.union(*surrounding_lanes_sets)

        # 2. For each vehicle, work out which lanes they're in
        vehicle_lanes = {v.vehicle_id: SafetyPureController.get_vehicle_lanes(v, road_network) for v in vehicles}
        if DEBUG:
            vehicle2 = [v for v in vehicles if v.vehicle_id[6] == "2"]
            vehicle2_id = None if len(vehicle2) == 0 else vehicle2[0].vehicle_id
            is_disjoint = None if len(vehicle2) == 0 else vehicle_lanes[vehicle2_id].isdisjoint(all_surrounding_lanes_set)

        # filter out vehicles not in the surrounding lanes
        vehicles = [v for v in vehicles if not vehicle_lanes[v.vehicle_id].isdisjoint(all_surrounding_lanes_set)]


        # 3. Order vehicles based on distance along ego vehicle's road
        lane_local_offsets = {v.vehicle_id: SafetyPureController.get_distance_into_lane(current_lane, v, road_network) for v in vehicles}
        vehicles.sort(key = lambda v : lane_local_offsets[v.vehicle_id])

        ego_pos = np.clip(road_network.offset_into_lane(current_lane, vehicle.pose.position[:2]), 0, current_lane.getLength())

        # Calculate the starting point/angles for the current lane
        start_point = road_network.world_coord_from_offset(current_lane, ego_pos)
        start_vector = road_network.lane_vector_at_offset(current_lane, ego_pos)
        start_heading = np.arctan2(start_vector[1], start_vector[0])


        # 4. Generate the surroundings and convert to CarsimCar
        surroundings = [[None, None] for i in range(len(surrounding_lanes))]
        for v in vehicles:
            for lane_num, surrounding_lane_set in enumerate(surrounding_lanes_sets):
                if not vehicle_lanes[v.vehicle_id].isdisjoint(surrounding_lane_set):
                    if lane_local_offsets[v.vehicle_id] < ego_pos:
                        # Car is in surrounding lane, and behind ego vehicle
                        surroundings[lane_num][0] = CarsimCar(v, start_point, start_heading)
                    else:
                        # Car is in surrounding lane, and ahead of ego vehicle
                        if surroundings[lane_num][1] is None:
                            surroundings[lane_num][1] = CarsimCar(v, start_point, start_heading)
        ego_car = CarsimCar(vehicle.state, start_point, start_heading)


        # 5. Work out the lane boundaries
        lane_bounds = []
        bound = 0
        centre = None
        for lane in surrounding_lanes:
            width = lane.getWidth()
            lane_bounds.append((bound, bound+width))
            if lane == current_lane:
                centre = bound + width / 2
            bound += width

        if centre is None:
            raise ValueError("Current lane is not in the surrounding lanes")

        for i, lb in enumerate(lane_bounds):
            lane_bounds[i] = (lb[0] - centre, lb[1] - centre)

        start_t = time.perf_counter()

        # 6. Pass through to our safety checker, finding an action which is hopefully close to the original intention,
        #    but definitely is safe.
        ego_car.a = a
        deltas = carsim.logic.get_safe_deltas(ego_car, lane_bounds, surroundings, dt)


        if DEBUG:
            changed = False

        # Check if deltas empty
        if deltas == S.EmptySet:
            a = - vehicle.max_brake
            ego_car.a = a
            deltas = carsim.logic.get_safe_deltas(ego_car, lane_bounds, surroundings, dt)

            if deltas == S.EmptySet:
                # There's no safe delta while max braking, so steer towards lane centre.
                delta =  -start_heading + vehicle.pose.heading + np.pi/2
            
        if deltas != S.EmptySet and not deltas.contains(delta):
            # Action is deemed unsafe, so find the closest possible delta, by extracting the boundary of 
            # the delta set, and picking the closest boundary
            boundary = deltas.boundary
            assert isinstance(boundary, sym.FiniteSet)
            new_delta = float(min(boundary.args, key=lambda d : abs(d-delta)))
            safety_thresh = 0.05
            if new_delta > delta:
                delta = new_delta + safety_thresh
            else:
                delta = new_delta - safety_thresh
            if DEBUG:
                changed = True

        if DEBUG:
            if vehicle.id[6] == "2":
                ax0.clear()
                carsim.plot.plot_surroundings_and_deltas(ego_car, surroundings, lane_bounds, dt, deltas, ax0)
                if changed:
                    ax0.text(0,0,"picked new delta = {:.2f}".format(delta))

                ax0.text(1, 1, "on road {}".format(current_lane.getID()))
            if vehicle.id[6] == "3":
                #print("info from vehicle id: {}".format(vehicle.id))
                #print("surroundings :")
                #for surrounding in surroundings:
                #    print("{}|{}".format(surrounding[0], surrounding[1]))

                ax1.clear()
                carsim.plot.plot_surroundings_and_deltas(ego_car, surroundings, lane_bounds, dt, deltas, ax1)
                if changed:
                    ax1.text(0,0,"picked new delta = {:.2f}".format(delta))
                #ax.text(0,0,"delta = {:.2f}".format(delta))
                surround_txt = "surrounding lanes : {}".format([[lane.getID() for lane in lane_list] for lane_list in surrounding_lanes_sets])
                on_road_txt = "veh is on {}, disjoint={}".format([l.getID() for l in vehicle_lanes[vehicle2_id]] if vehicle2_id is not None else None, is_disjoint)
                ax1.text(2,2, surround_txt)
                ax1.text(2,1, on_road_txt)
                ax1.text(0,0, "{}".format(lane_local_offsets))

                plt.pause(0.01)

                # print("lane_bounds = {}".format(lane_bounds))

    #            print("current_lane id : {}".format(current_lane.getID()))
    #            print("surrounding lanes : {}".format([[lane.getID() for lane in lane_list] for lane_list in surrounding_lanes_sets]))
    #            #print("vehicles : {}".format(vehicles))
    #            print("")
    #            for v in vehicles:
    #                #print("vehicle {} in lanes : {}".format(v.vehicle_id, [lane.getID() for lane in vehicle_lanes[v.vehicle_id]]))
    #                print("vehicle {} in local position : {}".format(v.vehicle_id, lane_local_offsets[v.vehicle_id]))


        if a > 0:
            brake = 0
            throttle = a / vehicle.max_accel
        else:
            brake = - a / vehicle.max_brake
            throttle = 0
        steering_angle = delta / (np.pi/4) 

        action = (throttle, brake, steering_angle)

        return action

    @staticmethod
    def get_front_bumper(vehicle_state):
        position = np.array(vehicle_state.pose.position[:2])
        theta = vehicle_state.pose.heading + np.pi/2
        position += np.array((np.cos(theta), np.sin(theta))) * vehicle_state.dimensions.length/2
        return position


    @staticmethod
    def get_distance_into_lane(lane, veh_state, road_network):
        offset = road_network.offset_into_lane(lane, veh_state.pose.position[:2])
        if offset == 0: # in case we are before the lane begins then find the euclidean distance from the start of the lane to use
            offset = - np.linalg.norm(road_network.world_coord_from_offset(lane, 0) - np.array(veh_state.pose.position[:2]))
        return offset


    @classmethod
    def perform_action(cls, sim, sensor_state, vehicle, action, dt):
        """
        Check if an action is safe, and if not modify it to be as safe as possible, then
        execute using pure physics (no pybullet).
        """
        action = SafetyPureController.get_action(cls, sim, sensor_state, vehicle, action, dt)
        PureController.perform_action(vehicle, action, dt)

    @staticmethod
    def get_vehicle_lanes(vehicle, road_network):
        x, y = vehicle.pose.position[:2]
        heading = vehicle.pose.heading + np.pi/2
        l = vehicle.dimensions.length
        w = vehicle.dimensions.width

        # Offsets are all the corners of the vehicle 
        # If some vehicles are not registering in a lane, may be worth adding more offsets
        offsets = np.array(((l/2, w/2), (l/2, -w/2), (-l/2, w/2), (-l/2, -w/2), (0, 0)))
        c, s = np.cos(heading), np.sin(heading)
        rot = np.array(((c, -s), (s, c)))
        offsets = np.array([rot.dot(offset) for offset in offsets])
        corners = np.array((x,y)) + offsets

        lanes = set()
        for corner in corners:
            lane = road_network.nearest_lane(corner)
            if lane is not None:
                lanes.add(lane)

        return lanes


class SafetyPureLaneFollowingController:
    @classmethod
    def perform_lane_following(
        cls,
        sim,
        agent_id,
        vehicle,
        sensor_state,
        dt,
        target_speed=12.5,
        lane_change=0,
    ):
        # For now this method simply passes the lane following action the safety controller, which should ensure that
        # each action is individually safe.
        #lane_change = 0
        action = PureLaneFollowingController.get_action(sim, vehicle, sensor_state, dt, target_speed, lane_change)
        SafetyPureController.perform_action(sim, sensor_state, vehicle, action, dt)

    @classmethod
    def get_lane_following(
        cls,
        sim,
        vehicle,
        sensor_state,
        dt,
        target_speed=12.5,
        lane_change=0,
    ):
        # For now this method simply passes the lane following action the safety controller, which should ensure that
        # each action is individually safe.
        #lane_change = 0
        action = PureLaneFollowingController.get_action(sim, vehicle, sensor_state, dt, target_speed, lane_change)
        action = SafetyPureController.get_action(sim, sensor_state, vehicle, action, dt)
        return action



