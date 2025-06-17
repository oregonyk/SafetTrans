from __future__ import print_function
import math
import itertools
import numpy.random as random

import carla
import py_trees
from carla import Transform, Location, Rotation
from agents.navigation.global_route_planner import GlobalRoutePlanner
from agents.navigation.global_route_planner_dao import GlobalRoutePlannerDAO

from srunner.scenarioconfigs.scenario_configuration import ScenarioConfiguration, ActorConfigurationData
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.weather_sim import WeatherBehavior
from srunner.scenarios.basic_scenario import BasicScenario
from srunner.scenariomanager.timer import GameTime
from srunner.scenariomanager.scenarioatomics.atomic_criteria import (CollisionTest,
                                                                     InRouteTest,
                                                                     RouteCompletionTest,
                                                                     OutsideRouteLanesTest,
                                                                     RunningRedLightTest,
                                                                     RunningStopTest,
                                                                     ActorSpeedAboveThresholdTest)

from leaderboard.scenarios.route_scenario import convert_transform_to_location, convert_json_to_transform, convert_json_to_transform
from leaderboard.utils.route_manipulation import location_route_to_gps, _get_latlon_ref
from leaderboard.utils.route_parser import RouteParser
from .route_scenario import RouteScenario

WEATHERS = {
    1: carla.WeatherParameters.ClearNoon,
    3: carla.WeatherParameters.WetNoon,
    6: carla.WeatherParameters.HardRainNoon,
    8: carla.WeatherParameters.ClearSunset,
    10: carla.WeatherParameters.WetSunset,
    14: carla.WeatherParameters.SoftRainSunset,
}

class NoCrashEvalScenario(RouteScenario):
    category = "NoCrashEvalScenario"
    
    def __init__(self, world, agent, start_idx, target_idx, weather_idx, traffic_idx, debug_mode=0, physics=[], criteria_enable=True):

        # Overwrite
        self.list_scenarios = []
        # pan
        # str_town_name = world.get_map().name
        # str_temp = str_town_name.split("/")
        self.town_name = world.get_map().name
        # self.town_name = str_temp[2]
        self.weather_idx = weather_idx
        self.start_idx = start_idx
        self.target_idx = target_idx
        self.traffic_idx = traffic_idx

        self.agent = agent

        self.physics = physics
        # Set route
        self._set_route()

        # set npc_id
        self.npc_id = None

        ego_vehicle = self._update_ego_vehicle(self.physics)
        traffic_lvl = ['Empty', 'Regular', 'Dense'][traffic_idx]
        
        BasicScenario.__init__(self, name=f'NoCrash_{self.town_name}_{traffic_idx}_w{weather_idx}_s{start_idx}_t{target_idx}',
            ego_vehicles=[ego_vehicle],
            config=None,
            world=world,
            debug_mode=debug_mode>1,
            terminate_on_failure=False,
            criteria_enable=criteria_enable
        )

        self.list_scenarios = []
        
    def _set_route(self, hop_resolution=1.0):

        world = CarlaDataProvider.get_world()
        dao = GlobalRoutePlannerDAO(world.get_map(), hop_resolution)
        grp = GlobalRoutePlanner(dao)
        grp.setup()

        spawn_points = CarlaDataProvider._spawn_points

        # pan !
        start = spawn_points[self.start_idx]
        target = spawn_points[self.target_idx]
        start = Transform(Location(x=324.010, y=198.3063, z=0.0333), Rotation(pitch=0.005, yaw=-179.2628, roll=0.000))
        target = Transform(Location(x=335.3134, y=220.4802, z=0.033), Rotation(pitch=0.000, yaw=90.3134, roll=-0.00))
        route = grp.trace_route(start.location, target.location)
        self.route = [(w.transform,c) for w, c in route]

        CarlaDataProvider.set_ego_vehicle_route([(w.transform.location, c) for w, c in route])
        gps_route = location_route_to_gps(self.route, *_get_latlon_ref(world))
        self.agent.set_global_plan(gps_route, self.route)

        self.timeout = self._estimate_route_timeout()
        
    def _initialize_actors(self, config):
        """
        Set other_actors to the superset of all scenario actors
        """
        # Create the background activity of the route
        car_amounts = {
            'Town01': [0, 20, 100],
            'Town02': [0, 15, 70]
        }
        
        ped_amounts = {
            'Town01': [0, 50, 200],
            'Town02': [0, 50, 150]
        }


        car_amount = car_amounts[self.town_name][self.traffic_idx]
        #print(car_amount)
        ped_amount = ped_amounts[self.town_name][self.traffic_idx]
        #print(ped_amount)

        new_actors = CarlaDataProvider.request_new_batch_actors('vehicle.*',
                                                                car_amount,
                                                                carla.Transform(),
                                                                autopilot=True,
                                                                random_location=True,
                                                                rolename='background')

        if new_actors is None:
            raise Exception("Error: Unable to add the background activity, all spawn points were occupied")

        blueprints = CarlaDataProvider._blueprint_library.filter('walker.pedestrian.*')
        spawn_points = []
        while len(spawn_points) < ped_amount:
        # < 1:
            spawn_point = carla.Transform()
            loc = CarlaDataProvider.get_world().get_random_location_from_navigation()
            if (loc != None):
                spawn_point.location = loc
                spawn_points.append(spawn_point)

        batch = []
        for spawn_point in spawn_points:
            walker_bp = random.choice(blueprints)
            if walker_bp.has_attribute('is_invincible'):
                walker_bp.set_attribute('is_invincible', 'false')
            batch.append(carla.command.SpawnActor(walker_bp, spawn_point))
        
        pedestrians = CarlaDataProvider.handle_actor_batch(batch)
        
        batch = []
        walker_controller_bp = CarlaDataProvider._blueprint_library.find('controller.ai.walker')
        for pedestrian in pedestrians:
            batch.append(
                carla.command.SpawnActor(walker_controller_bp, carla.Transform(), pedestrian)
            )

        pedestrian_controllers = CarlaDataProvider.handle_actor_batch(batch)
        CarlaDataProvider.get_world().set_pedestrians_cross_factor(1.0)
        for controller in pedestrian_controllers:
            controller.start()
            controller.go_to_location(CarlaDataProvider.get_world().get_random_location_from_navigation())
            controller.set_max_speed(1.2 + random.random())
            
        for actor in itertools.chain(pedestrians, pedestrian_controllers):
            if actor is None:
                continue

            CarlaDataProvider._carla_actor_pool[actor.id] = actor
            CarlaDataProvider.register_actor(actor)

            self.other_actors.append(actor)
        batch = []
        # add NPC vehicle
        if self.town_name == 'Town01' and car_amount == 0 and ped_amount == 0:
            # vehicle_bp = CarlaDataProvider._blueprint_library.find('vehicle.tesla.model3')
            walker_bp = random.choice(CarlaDataProvider._blueprint_library.filter('walker.pedestrian.0016'))
            if walker_bp.has_attribute('is_invincible'):
                walker_bp.set_attribute('is_invincible', 'false')
            # spawn_point_one = Transform(Location(x=200.670166, y=195.148956, z=0.30000), Rotation(pitch=0, yaw=179.999756, roll=0))
            #NOTE walker position need to be change along with scenario_manager.py pos
            walker_x = 338.670166
            walker_y = 211.048956
            spawn_point_one = Transform(Location(x=walker_x, y=walker_y, z=0.30000))
            CarlaDataProvider.get_world().debug.draw_string(Location(x=walker_x, y=walker_y, z=0.30000), '0', draw_shadow=False,
                                                            color=carla.Color(r=0, g=255, b=0), life_time=100000,
                                                            persistent_lines=True)
            CarlaDataProvider.get_world().debug.draw_string(Location(x=210.670166, y=190.148956, z=0.30000), '0',
                                                            draw_shadow=False,
                                                            color=carla.Color(r=255, g=255, b=255), life_time=100000,
                                                            persistent_lines=True)
            CarlaDataProvider.get_world().debug.draw_string(Location(x=210.670166, y=200.148956, z=0.30000), '0',
                                                            draw_shadow=False,
                                                            color=carla.Color(r=255, g=0, b=0), life_time=100000,
                                                            persistent_lines=True)
            # vehicle_one = CarlaDataProvider.get_world().spawn_actor(vehicle_bp, spawn_point_one)
            # batch.append(carla.command.SpawnActor(vehicle_bp, spawn_point_one))
            batch.append(carla.command.SpawnActor(walker_bp, spawn_point_one))

            vehicle_one = CarlaDataProvider.handle_actor_batch(batch)

            for vehicle in vehicle_one:
                CarlaDataProvider._carla_actor_pool[vehicle.id] = vehicle
                CarlaDataProvider.register_actor(vehicle)
                self.other_actors.append(vehicle)
                self.npc_id = vehicle.id
                # print("npc_id:", vehicle.type_id)

    def _initialize_environment(self, world):

        world.set_weather(WEATHERS[self.weather_idx])
        
    def _setup_scenario_trigger(self, config):
        pass
    
    def _setup_scenario_end(self, config):
        """
        This function adds and additional behavior to the scenario, which is triggered
        after it has ended.

        The function can be overloaded by a user implementation inside the user-defined scenario class.
        """
        pass
    
    def _create_test_criteria(self):
        """
        """
        criteria = []
        route = convert_transform_to_location(self.route)

        collision_criterion = CollisionTest(self.ego_vehicles[0], terminate_on_failure=False)

        route_criterion = InRouteTest(self.ego_vehicles[0],
                                      route=route,
                                      offroad_max=30,
                                      terminate_on_failure=True)
                                      
        completion_criterion = RouteCompletionTest(self.ego_vehicles[0], route=route)

        outsidelane_criterion = OutsideRouteLanesTest(self.ego_vehicles[0], route=route)

        red_light_criterion = RunningRedLightTest(self.ego_vehicles[0])

        stop_criterion = RunningStopTest(self.ego_vehicles[0])

        blocked_criterion = ActorSpeedAboveThresholdTest(self.ego_vehicles[0],
                                                         speed_threshold=0.1,
                                                         below_threshold_max_time=180.0,
                                                         terminate_on_failure=True,
                                                         name="AgentBlockedTest")

        criteria.append(completion_criterion)
        criteria.append(outsidelane_criterion)
        criteria.append(collision_criterion)
        criteria.append(red_light_criterion)
        criteria.append(stop_criterion)
        criteria.append(route_criterion)
        criteria.append(blocked_criterion)

        return criteria