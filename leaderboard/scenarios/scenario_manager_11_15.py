#!/usr/bin/env python

# Copyright (c) 2018-2020 Intel Corporation
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
This module provides the ScenarioManager implementations.
It must not be modified and is for reference only!
"""

from __future__ import print_function
import signal
import sys
import time

import py_trees
import carla
import math
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.timer import GameTime
from srunner.scenariomanager.watchdog import Watchdog

from leaderboard.autoagents.agent_wrapper import AgentWrapper, AgentError
from leaderboard.envs.sensor_interface import SensorReceivedNoData
from leaderboard.utils.result_writer import ResultOutputProvider
from carla import Vector3D
from srunner.scenariomanager.carla_data_provider import calculate_velocity


class ScenarioManager(object):
    """
    Basic scenario manager class. This class holds all functionality
    required to start, run and stop a scenario.

    The user must not modify this class.

    To use the ScenarioManager:
    1. Create an object via manager = ScenarioManager()
    2. Load a scenario via manager.load_scenario()
    3. Trigger the execution of the scenario manager.run_scenario()
       This function is designed to explicitly control start and end of
       the scenario execution
    4. If needed, cleanup with manager.stop_scenario()
    """

    def __init__(self, timeout, debug_mode=False):
        """
        Setups up the parameters, which will be filled at load_scenario()
        """
        self.scenario = None
        self.scenario_tree = None
        self.scenario_class = None
        self.ego_vehicles = None
        self.other_actors = None

        self._debug_mode = debug_mode
        self._agent = None
        self._running = False
        self._timestamp_last_run = 0.0
        self._timeout = float(timeout)
        # --------------pan---------------
        self._pan = 0
        self.npc_id = None
        self.revert_flag = False
        self.flag_count = 0
        self.distance = 0
        self.overTTC = []
        self.overDRAC = []
        self.t1 = []
        self.t2 = []
        self.index = 0
        self.dac = []
        self.TET = 0
        self.TIT = 0
        self.average_dacc = 0
        # self.record_flag=False
        # --------------------------------

        # Used to detect if the simulation is down
        watchdog_timeout = max(5, self._timeout - 2)
        self._watchdog = Watchdog(watchdog_timeout)

        # Avoid the agent from freezing the simulation
        agent_timeout = watchdog_timeout - 1
        self._agent_watchdog = Watchdog(agent_timeout)

        self.scenario_duration_system = 0.0
        self.scenario_duration_game = 0.0
        self.start_system_time = None
        self.end_system_time = None
        self.end_game_time = None

        # Register the scenario tick as callback for the CARLA world
        # Use the callback_id inside the signal handler to allow external interrupts
        signal.signal(signal.SIGINT, self.signal_handler)

    def signal_handler(self, signum, frame):
        """
        Terminate scenario ticking when receiving a signal interrupt
        """
        self._running = False

    def cleanup(self):
        """
        Reset all parameters
        """
        self._timestamp_last_run = 0.0
        self.scenario_duration_system = 0.0
        self.scenario_duration_game = 0.0
        self.start_system_time = None
        self.end_system_time = None
        self.end_game_time = None

    def load_scenario(self, scenario, agent, rep_number):
        """
        Load a new scenario
        """

        GameTime.restart()
        self._agent = AgentWrapper(agent)
        self.scenario_class = scenario
        self.scenario = scenario.scenario
        self.scenario_tree = self.scenario.scenario_tree
        self.ego_vehicles = scenario.ego_vehicles
        self.other_actors = scenario.other_actors
        self.repetition_number = rep_number

        # set npc_id
        self.npc_id = scenario.npc_id

        # To print the scenario tree uncomment the next line
        # py_trees.display.render_dot_tree(self.scenario_tree)

        self._agent.setup_sensors(self.ego_vehicles[0], self._debug_mode)

    def run_scenario(self):
        """
        Trigger the start of the scenario and wait for it to finish/fail
        """
        self.start_system_time = time.time()
        self.start_game_time = GameTime.get_time()

        self._watchdog.start()
        self._running = True

        while self._running:
            timestamp = None
            world = CarlaDataProvider.get_world()
            if world:
                snapshot = world.get_snapshot()
                if snapshot:
                    timestamp = snapshot.timestamp
            if timestamp:
                self._tick_scenario(timestamp)

        # print("num TTC ->", len(self.overTTC), "--> DTAC ->", len(self.overDRAC))
        # print("TIT :", self.call_TIT(), " TET :", self.call_TET())
        # sum = 0
        #
        # for i in range(len(self.dac)):
        #     sum = sum + abs(self.dac[i])
        #
        # print("average:", sum / len(self.dac))

    def cal_speed(self, actor):
        velocity_squared = actor.get_velocity().x ** 2
        velocity_squared += actor.get_velocity().y ** 2
        return math.sqrt(velocity_squared)

    def cal_rela_loc(self, actor, pes):
        loc_sq = (actor.get_location().x - pes.get_location().x) ** 2
        loc_sq += (actor.get_location().y - pes.get_location().y) ** 2
        return math.sqrt(loc_sq)

    def cal_rela_speed(self, actor, pes):
        current_dis = actor.get_location().x - 210.670166
        rela_loc = self.cal_rela_loc(actor, pes)
        cos_rate = current_dis / rela_loc
        # print("cos:",cos_rate)
        v_a = self.cal_speed(actor) * cos_rate
        v_p = self.cal_speed(pes) * math.sqrt(1 - (cos_rate ** 2))
        return v_a + v_p

    def call_TTC(self, actor, pes):
        loc = self.cal_rela_loc(actor, pes)
        velocity = self.cal_rela_speed(actor, pes)
        TTC = (loc - 2.4508) / velocity
        # TTC = (loc - 2.6719) / velocity
        TTC = float('%.3f' % TTC)
        return TTC

    def call_DRAC(self, actor, pes):
        velocity = self.cal_rela_speed(actor, pes)
        loc = self.cal_rela_loc(actor, pes)
        DRAC = (velocity ** 2) / (loc - 2.4508)
        DRAC = float('%.3f' % DRAC)
        return DRAC

    def call_TET(self):
        TET = len(self.overTTC) * 0.05
        return float('%.3f' % TET)

    def call_TIT(self):
        TIT = 0
        for i in range(len(self.overTTC)):
            index = 1.5 - self.overTTC[i]
            TIT = TIT + (index * 0.05)
        return float('%.3f' % TIT)

    def _tick_scenario(self, timestamp):
        """
        Run next tick of scenario and the agent and tick the world.
        """

        if self._timestamp_last_run < timestamp.elapsed_seconds and self._running:
            self._timestamp_last_run = timestamp.elapsed_seconds

            self._watchdog.update()
            # Update game time and actor information
            GameTime.on_carla_tick(timestamp)
            CarlaDataProvider.on_carla_tick()

            try:
                ego_action = self._agent()

            # Special exception inside the agent that isn't caused by the agent
            except SensorReceivedNoData as e:
                raise RuntimeError(e)

            except Exception as e:
                raise AgentError(e)

            self.ego_vehicles[0].apply_control(ego_action)

            if ego_action.brake == 1.0:
                if abs(self.ego_vehicles[0].get_velocity().x) > 0.001:
                    self.dac.append(float('%.3f' % self.ego_vehicles[0].get_acceleration().x))

            # Tick scenario
            self.scenario_tree.tick_once()

            if self._debug_mode:
                print("\n")
                py_trees.display.print_ascii_tree(
                    self.scenario_tree, show_status=True)
                sys.stdout.flush()

            if self.scenario_tree.status != py_trees.common.Status.RUNNING:
                self._running = False
            ego_trans = self.ego_vehicles[0].get_transform()
            # top view : every step

            if self._pan < 50:
                spectator = CarlaDataProvider.get_world().get_spectator()
                ego_trans = self.ego_vehicles[0].get_transform()
                spectator.set_transform(carla.Transform(ego_trans.location + carla.Location(z=50),
                                                        carla.Rotation(pitch=-90)))
                CarlaDataProvider.get_world().debug.draw_string(ego_trans.location, '0', draw_shadow=False,
                                                                color=carla.Color(r=255, g=0, b=0), life_time=100000,
                                                                persistent_lines=True)
            # print(self.ego_vehicles[0].get_physics_control())
            # car head before npc we collect ttc ,if reached do not collect
            #by mjw 2023.11.10
            '''if ego_trans.location.x >= 212:
                if float('%.3f' % self.ego_vehicles[0].get_velocity().x) != 0:
                    temp = ego_trans.location.x - 210
                    # drac = float('%.3f' % self.ego_vehicles[0].get_velocity().x) * float(
                    #     '%.3f' % self.ego_vehicles[0].get_velocity().x) / float('%.3f' % temp)
                    npc = CarlaDataProvider._carla_actor_pool[self.npc_id]
                    ttc = self.call_TTC(self.ego_vehicles[0], npc)
                    drac = self.call_DRAC(self.ego_vehicles[0], npc)
                    ttc = float('%.3f' % ttc)
                    drac = float('%.3f' % drac)
                    # print("drac:", abs(drac), "ttc:", abs(ttc))
                    if abs(ttc) <= 1.5:
                        self.t1.append(ttc)
                    if abs(drac) >= 3.45:
                        self.t2.append(drac)
            # when car stop we stop ttc ,
            if 0.001 > self.ego_vehicles[0].get_velocity().x >= 0:
                self.distance = ego_trans.location.x - 210.670166 - 2.4508
                # 1. car already run 2. record once
                if ego_trans.location.x < 220 and self.index == 0:
                    # print("write")
                    for i in range(len(self.t1)):
                        self.overTTC.append(self.t1[i])
                    for i in range(len(self.t2)):
                        self.overDRAC.append(self.t2[i])
                    self.index = 1
'''
            ttc = 0.111
            drac = 0.111
            # tick count
            # print("time:",GameTime.get_time())
            # print(self.ego_vehicles[0].get_location())
            self._pan = self._pan + 1
            if self.npc_id is not None:
                npc = CarlaDataProvider._carla_actor_pool[self.npc_id]
                # print(npc.get_location())
                if npc is not None:
                    control = carla.VehicleControl()
                    '''control = carla.WalkerControl()
                    control.direction.x = 0
                    control.direction.z = 0
                    control.speed = 1.8
                    # print(npc.get_location())
                    if npc.get_location().y > 196:  # 我是提前知道了大概转头的y位置

                        self.revert_flag = True  # 到了就转头
                    if npc.get_location().y < 190:
                        self.revert_flag = False
                    if self.revert_flag:
                        control.direction.y = -1
                    else:
                        control.direction.y = 1'''

                    control.speed = 50.0
                    npc.set_autopilot(True)
                    #npc.apply_control(carla.VehicleControl(throttle=0.6, steer=0.0))
            if GameTime.get_time() > 10:
            #mjw
                try:
                    npc.set_autopilot(False)
                except:

                    pass
            #if GameTime.get_time() > 9.5:
                self._running = False


        if self._running and self.get_running_status():
            CarlaDataProvider.get_world().tick(self._timeout)
            # print("1")

    def get_running_status(self):
        """
        returns:
           bool: False if watchdog exception occured, True otherwise
        """
        return self._watchdog.get_status()

    def stop_scenario(self):
        """
        This function triggers a proper termination of a scenario
        """
        self._watchdog.stop()

        self.end_system_time = time.time()
        self.end_game_time = GameTime.get_time()

        self.scenario_duration_system = self.end_system_time - self.start_system_time
        self.scenario_duration_game = self.end_game_time - self.start_game_time

        self.TET = self.call_TET()
        self.TIT = self.call_TIT()
        total_sum = 0
        for i in range(len(self.dac)):
            total_sum = total_sum + abs(self.dac[i])
        # print(self.overTTC)
        #mjw for scenario building
        if len(self.dac) == 0:
            self.average_dacc = total_sum
        else:
            self.average_dacc = float('%.3f' % (total_sum / len(self.dac)))
        #self.average_dacc = 1.0
        # pan
        self._pan = 0
        self.overTTC = []
        self.overDRAC = []
        self.t1 = []
        self.t2 = []
        self.index = 0
        self.dac = []

        if self.get_running_status():
            if self.scenario is not None:
                self.scenario.terminate()

            if self._agent is not None:
                self._agent.cleanup()
                self._agent = None

            self.analyze_scenario()

    def analyze_scenario(self):
        """
        Analyzes and prints the results of the route
        """
        global_result = '\033[92m' + 'SUCCESS' + '\033[0m'

        for criterion in self.scenario.get_criteria():
            if criterion.test_status != "SUCCESS":
                global_result = '\033[91m' + 'FAILURE' + '\033[0m'

        if self.scenario.timeout_node.timeout:
            global_result = '\033[91m' + 'FAILURE' + '\033[0m'

        ResultOutputProvider(self, global_result)

    def get_nocrash_diagnostics(self):

        route_completion = None
        lights_ran = None
        duration = round(self.scenario_duration_game, 2)

        for criterion in self.scenario.get_criteria():
            actual_value = criterion.actual_value
            name = criterion.name

            if name == 'RouteCompletionTest':
                route_completion = float(actual_value)
            elif name == 'RunningRedLightTest':
                lights_ran = int(actual_value)

        return route_completion, lights_ran, duration

    def get_nocrash_objective_data(self):
        for criterion in self.scenario.get_criteria():
            name = criterion.name
            if name == 'CollisionTest':
                if criterion.speed is not None:
                    if criterion.speed > 0:
                        speed = - criterion.speed
                        return speed
                    else:
                        return criterion.speed
                if criterion.actual_value == 0:
                    return self.distance

    def get_nocrash_analyze_data(self):
        return self.TET, self.TIT, self.average_dacc
