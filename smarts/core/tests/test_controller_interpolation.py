# MIT License
#
# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
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
import pytest

import numpy as np
import smarts.sstudio.types as t
from smarts.core.agent import AgentSpec, Agent
from smarts.core.agent_interface import AgentInterface, AgentType
from smarts.core.controllers import LaneFollowingController
from smarts.core.scenario import Scenario
from smarts.core.smarts import SMARTS
from smarts.core.sumo_traffic_simulation import SumoTrafficSimulation
from smarts.core.tests.helpers.scenario import temp_scenario
from smarts.sstudio import gen_scenario

AGENT_ID = "Agent-007"

# Tests are parameterized based on different with-time-trajectory
@pytest.fixture(
    # Each parameter item is (Agent will, with-time-trajectory)
    params=[
        # Test illegal input
        (
            "Stop",
            np.array([]),
        ),
        (
            "Stop",
            np.array(
                [
                    [0.0],  # TIME
                    [100.0],  # X
                    [2.0],  # Y
                    [3.0],  # THETA
                    [4.0],  # VEL
                ]
            ),
        ),
        (
            "Stop",
            np.array(
                [
                    [0.0, 0.2],  # TIME
                    [12.0, 100.0],  # X
                    [2.0, np.nan],  # Y
                    [3.0, np.inf],  # THETA
                    [4.0, 400.0],  # VEL
                ]
            ),
        ),
        (
            "Stop",
            np.array(
                [
                    [1.0, 2.0, 3.0],  # TIME. Can not locate motion state.
                    [1.0, 2.0, 3.0],  # X
                    [2.0, 3.0, 4.0],  # Y
                    [3.0, 4.0, 5.0],  # THETA
                    [4.0, 5.0, 6.0],  # VEL
                ]
            ),
        ),
        # Test trajectory with different time resolution.
        (
            "Goal",
            np.array(
                [
                    [
                        0.0,
                        0.2,
                        0.3,
                    ],  # TIME. Resolution is greater than SMARTS timestep.
                    [1.0, 20.0, 30.0],  # X
                    [0.0, 0.0, 0.0],  # Y
                    [0.0, 0.0, 0.0],  # THETA
                    [4.0, 4.0, 4.0],  # VEL
                ]
            ),
        ),
        (
            "Goal",
            np.array(
                [
                    [
                        0.0,
                        0.05,
                        0.1,
                        0.15,
                        0.2,
                    ],  # TIME. Resolution is smaller than SMARTS timestep.
                    [1.0, 2.0, 10.0, 200.0, 300.0],  # X
                    [0.0, 0.0, 0.0, 0.0, 0.0],  # Y
                    [0.0, 0.0, 0.0, 0.0, 0.0],  # THETA
                    [4.0, 4.0, 4.0, 4.0, 4.0],  # VEL
                ]
            ),
        ),
        (
            "Goal",
            np.array(
                [
                    [0.0, 0.05, 0.2],  # TIME. Arbitary time interval.
                    [1.0, 2.0, 100.0],  # X
                    [0.0, 0.0, 0.0],  # Y
                    [0.0, 0.0, 0.0],  # THETA
                    [1.0, 1.0, 1.0],  # VEL
                ]
            ),
        ),
    ]
)
def case(request):
    return request.param


class WithTimeTrajectoryAgent(Agent):
    def __init__(self, will, local_traj):
        self._will = will
        self._local_traj = local_traj

    def act(self, obs):
        curr_position = obs.ego_vehicle_state.position
        curr_heading = obs.ego_vehicle_state.heading
        curr_speed = obs.ego_vehicle_state.speed
        new_origin_state = np.array(
            [
                [0.0],
                [curr_position[0]],
                [curr_position[1]],
                [curr_heading],
                [curr_speed],
            ]
        )
        return new_origin_state + self._local_traj

    @property
    def will(self):
        return self._will


@pytest.fixture()
def scenarios():
    with temp_scenario(name="map", map="maps/straight.net.xml") as scenario_root:
        mission = t.Mission(
            route=t.Route(
                begin=("west", 1, 100),
                end=("east", 1, 10.0),
            )
        )
        gen_scenario(
            t.Scenario(ego_missions=[mission]),
            output_dir=scenario_root,
        )
        yield Scenario.variations_for_all_scenario_roots(
            [str(scenario_root)], [AGENT_ID]
        )


@pytest.fixture
def agent_spec(case):
    for agent in AgentType:
        print(agent)
    return AgentSpec(
        interface=AgentInterface.from_type(
            AgentType.TrajectoryInterpolator, neighborhood_vehicles=True
        ),
        agent_builder=WithTimeTrajectoryAgent,
        agent_params=case,
    )


@pytest.fixture
def smarts(agent_spec):
    smarts = SMARTS(
        agent_interfaces={AGENT_ID: agent_spec.interface},
        traffic_sim=SumoTrafficSimulation(),
        timestep_sec=0.1,
    )
    yield smarts
    smarts.destroy()


def test_trajectory_interpolation_controller(smarts, agent_spec, scenarios):
    """Test trajectory interpolation controller

    With different planning algorithm of WithTimeTrajectoryAgent,
    vehicle is going to accomplish its mission or not.

    """
    agent = agent_spec.build_agent()
    scenario = next(scenarios)
    observations = smarts.reset(scenario)
    init_ego_state = observations[AGENT_ID].ego_vehicle_state

    agent_obs = None
    reached_goal = False
    for _ in range(50):
        agent_obs = observations[AGENT_ID]
        agent_action = agent.act(agent_obs)
        try:
            observations, _, dones, _ = smarts.step({AGENT_ID: agent_action})
        except Exception:
            continue

        if agent_obs.events.reached_goal:
            reached_goal = True
            break

    if agent.will == "Stop":
        assert (
            agent_obs.ego_vehicle_state == init_ego_state
        ), "Agent failed to stand still."
    elif agent.will == "Goal":
        assert reached_goal, "Agent failed to accomplish its mission."
    else:
        assert False, "Illegal agent will"
