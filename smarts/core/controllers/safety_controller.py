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
import carsim.plot

import sympy as sym
from sympy import S

import matplotlib.pyplot as plt

DEBUG = True
if DEBUG:
    ax = plt.gca()
    ax.set_aspect("equal")


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



class SafetyPureController:
    @classmethod
    def perform_action(cls, sim, sensor_state, vehicle, action, dt): # TODO: change the call for this to match
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

        road_network = sensor_state.mission_planner._road_network

        # Calculate safe actions
        # 1. Get all surrounding vehicles
        vehicles = sim.neighborhood_vehicles_around_vehicle(vehicle = vehicle, radius = np.inf) 

        # 2. Calculate ego vehicle surrounding lanes list
        current_lane = road_network.nearest_lane(vehicle.pose.position[:2])
        surrounding_lanes = current_lane.getEdge().getLanes()
        surrounding_lanes_sets = [{*lane.getIncoming(), lane} for lane in surrounding_lanes] # extend with road predecessors
        all_surrounding_lanes_set = set.union(*surrounding_lanes_sets)

        # 3. For each vehicle, work out which lanes they're in
        vehicle_lanes = {v.vehicle_id: SafetyPureController.get_vehicle_lanes(v, road_network) for v in vehicles}
        # filter out vehicles not in the surrounding lanes
        vehicles = [v for v in vehicles if not vehicle_lanes[v.vehicle_id].isdisjoint(all_surrounding_lanes_set)]


        # 4. Order vehicles based on distance along ego vehicle's road
        lane_local_offsets = {v.vehicle_id: road_network.offset_into_lane(current_lane, v.pose.position[:2]) for v in vehicles}
        vehicles.sort(key = lambda v : lane_local_offsets[v.vehicle_id])

        ego_pos = road_network.offset_into_lane(current_lane, vehicle.pose.position[:2])

        # Calculate the starting point/angles for the current lane
        start_point = road_network.world_coord_from_offset(current_lane, ego_pos)
        start_vector = road_network.lane_vector_at_offset(current_lane, ego_pos)
        start_heading = np.arctan2(start_vector[1], start_vector[0])


        # 5. Generate the surroundings and convert to CarsimCar
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


        # 6. Work out the lane boundaries
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

        # 7. Pass through to our safety checker, finding an action which is hopefully close to the original intention,
        #    but definitely is safe.
        ego_car.a = a
        deltas = carsim.logic.get_safe_deltas(ego_car, lane_bounds, surroundings, dt)

        # Check if deltas empty
        
        if deltas == S.EmptySet:
            print("no safe delta with accel = {}".format(a))
            a = - vehicle.max_brake
            ego_car.a = a
            deltas = carsim.logic.get_safe_deltas(ego_car, lane_bounds, surroundings, dt)

        if deltas == S.EmptySet:
            print("no safe delta with accel = 0")
            # There's no safe delta while max braking, so steer towards lane centre.
            if start_heading > vehicle.pose.heading:
                delta = np.pi / 4
            elif start_heading > vehicle.pose.heading:
                delta = - np.pi / 4
            else:
                delta = 0
            
        if DEBUG:
            changed = False
        else:
            if not deltas.contains(delta):
                # Action is deemed unsafe, so find the closest possible delta
                boundary = deltas.boundary
                assert isinstance(boundary, sym.FiniteSet)
                delta = min(boundary.args, key=lambda d : abs(d-delta))
                if DEBUG:
                    changed = True

        if DEBUG:
            if vehicle.id[6] == "2":
                print("info from vehicle id: {}".format(vehicle.id))
                print("surroundings :")
                for surrounding in surroundings:
                    print("{}|{}".format(surrounding[0], surrounding[1]))
                if changed:
                    ax.text(0,0,"picked new delta = {:.2f}".format(delta))

                ax.clear()
                carsim.plot.plot_surroundings_and_deltas(ego_car, surroundings, lane_bounds, dt, deltas, ax)
                #ax.text(0,0,"delta = {:.2f}".format(delta))

                plt.pause(0.1)

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
        PureController.perform_action(vehicle, action, dt)



    @staticmethod
    def get_vehicle_lanes(vehicle, road_network):
        x, y = vehicle.pose.position[:2]
        heading = vehicle.pose.heading + np.pi/2
        l = vehicle.dimensions.length
        w = vehicle.dimensions.width

        # Offsets are all the corners of the vehicle 
        # If some vehicles are not registering in a lane, may be worth adding more offsets
        offsets = np.array(((l/2, w/2), (l/2, -w/2), (-l/2, w/2), (-l/2, -w/2)))
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
        controller_state,
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

