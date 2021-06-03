import numpy as np
from smarts.core.controllers.safety_controller import CarsimCar

class Lane:
    def __init__(self, road_network, sumo_lane, start_x, centre_y, parallel_lanes):
        self.road_network = road_network
        self.sumo_lane = sumo_lane
        self.start_x = start_x
        self.centre_y = centre_y
        self.parallel_lanes = parallel_lanes


    def get_vehicle_position(self, veh_state):
        position = veh_state.pose.position[:2]
        u, v = self.road_network.world_to_lane_coord(self.sumo_lane, position)

        lane_vector = self.road_network.lane_vector_at_offset(self.sumo_lane, u)
        lane_heading = np.arctan2(lane_vector[1], lane_vector[0])
        theta = (veh_state.pose.heading + np.pi/2) - lane_heading

        x = u + self.start_x
        y = v + self.centre_y

        return x, y, theta

    def get_bound(self):
        width = self.sumo_lane.getWidth()
        return (self.centre_y - width/2, self.centre_y + width/2)


class LaneGroup:
    def __init__(self):
        self.lanes = []

    def add_lane(self, lane):
        if lane not in self.lanes:
            self.lanes.append(lane)

    @property
    def sumo_lanes(self):
        return [lane.sumo_lane.getID() for lane in self.lanes]



class SafetyRoadNetwork:
    def __init__(self, road_network):
        self.road_network = road_network
        self.lanes = {}
        lane = road_network.graph.getEdges()[0].getLanes()[0]
        self.get_lanes(lane)


    def get_lanes(self, start_lane):
        covered = []
        def recurse(lane, start_x, centre_y, lane_group=None):
            if lane in covered:
                return

            if lane_group is None:
                lane_group = LaneGroup()

            covered.append(lane)
            self.lanes[lane.getID()] = Lane(self.road_network, lane, start_x, centre_y, lane_group)
            lane_group.add_lane(self.lanes[lane.getID()])

            front = lane.getOutgoing()
            if len(front) > 0:
                front = front[0].getToLane()
                recurse(front, start_x + lane.getLength(), centre_y, lane_group)

            rear = lane.getIncoming(onlyDirect=True)
            if len(rear) > 0:
                rear = rear[0]
                recurse(rear, start_x - rear.getLength(), centre_y, lane_group)

            surrounding_lanes = lane.getEdge().getLanes()
            if len(surrounding_lanes) > 1:
                idx = surrounding_lanes.index(lane)
                if idx > 1:
                    right = surrounding_lane[idx-1]
                    recurse(right, start_x, centre_y - lane.getWidth()/2 - right.getWidth()/2)

                if idx < len(surrounding_lanes) - 1:
                    left = surrounding_lanes[idx + 1]
                    recurse(left, start_x, centre_y + lane.getWidth()/2 + left.getWidth()/2)

        recurse(start_lane, 0, 0)


    def get_carsim_car(self, vehicle_state, a=None):
        lane = self.lanes[self.road_network.nearest_lane(vehicle_state.pose.position[:2], include_junctions=False).getID()]

        x, y, theta = lane.get_vehicle_position(vehicle_state)

        return CarsimCar(vehicle_state, x, y, theta, a)



    
        
