from typing import Dict, Tuple

from gym.envs.registration import register
import numpy as np

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv, MultiAgentWrapper
from highway_env.road.lane import LineType, StraightLane, CircularLane, AbstractLane
from highway_env.road.regulation import RegulatedRoad
from highway_env.road.road import RoadNetwork
from highway_env.vehicle.kinematics import Vehicle
from highway_env.vehicle.controller import ControlledVehicle, MDPNoColVehicle
from highway_env.vehicle.behavior import IDMNoColVehicle, IDMBaseLineVehicle


class IntersectionNewEnv(AbstractEnv):

    ACTIONS: Dict[int, str] = {
        0: 'SLOWER',
        1: 'IDLE',
        2: 'FASTER'
    }
    ACTIONS_INDEXES = {v: k for k, v in ACTIONS.items()}

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            "observation": {
                "type": "Kinematics",
                "vehicles_count": 15,
                "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
                "features_range": {
                    "x": [-100, 100],
                    "y": [-100, 100],
                    "vx": [-20, 20],
                    "vy": [-20, 20],
                },
                "absolute": True,
                "flatten": False,
                "observe_intentions": True
            },
            "action": {
                "type": "DiscreteMetaAction",
                "longitudinal": True,
                "lateral": False
            },
            "duration": 30,  # [s]
            "destination": "o1",
            "controlled_vehicles": 1,
            "initial_vehicle_count": 5,
            "spawn_probability": 0.1,
            "screen_width": 960,
            "screen_height": 960,
            "centering_position": [0.5, 0.6],
            "scaling": 16, #5.5 * 1.3,
            "collision_reward": -10,
            "high_speed_reward": 1,
            "arrived_reward": 5,
            "reward_speed_range": [7.0, 9.0],
            "normalize_reward": False,
            "policy_frequency": 15,
            "simulation_frequency": 15,
            "max_vehicle_count": 5
        })
        return config

    def _reward(self, action: int) -> float:
        # Cooperative multi-agent reward
        return sum(self._agent_reward(action, vehicle) for vehicle in self.controlled_vehicles) \
               / len(self.controlled_vehicles)

    def _agent_reward(self, action: int, vehicle: Vehicle) -> float:
        scaled_speed = utils.lmap(self.vehicle.speed, self.config["reward_speed_range"], [0, 1])
        # if vehicle.CURRENT_CRASH:
        # crash_ratio = 1 if vehicle.CURRENT_DIST < vehicle.WIDTH+1 else -0.5 * vehicle.CURRENT_DIST + 2.5
        crash_ratio = 1

        # reward = self.config["collision_reward"] * vehicle.CURRENT_CRASH * crash_ratio\
        #          + self.config["high_speed_reward"] * np.clip(scaled_speed, 0, 1)
        if self.config["no_speed_reward"]:
            reward = self.config["collision_reward"] * vehicle.crashed * crash_ratio + self.config["step_reward"]
        else:
            reward = self.config["collision_reward"] * vehicle.crashed * crash_ratio \
                     + self.config["high_speed_reward"] * np.clip(scaled_speed, 0, 1)

        reward = self.config["arrived_reward"] if self.has_arrived(vehicle) else reward
        if self.config["normalize_reward"]:
            reward = utils.lmap(reward, [self.config["collision_reward"], self.config["arrived_reward"]], [0, 1])
        return reward

    def _is_terminal(self) -> bool:
        return any(vehicle.crashed for vehicle in self.controlled_vehicles) \
               or all(self.has_arrived(vehicle) for vehicle in self.controlled_vehicles) \
               or self.steps >= self.config["duration"] * self.config["policy_frequency"]

    def _agent_is_terminal(self, vehicle: Vehicle) -> bool:
        """The episode is over when a collision occurs or when the access ramp has been passed."""
        return vehicle.crashed \
            or self.steps >= self.config["duration"] * self.config["policy_frequency"] \
            or self.has_arrived(vehicle)

    def _info(self, obs: np.ndarray, action: int) -> dict:
        info = super()._info(obs, action)
        info["agents_rewards"] = tuple(self._agent_reward(action, vehicle) for vehicle in self.controlled_vehicles)
        info["agents_dones"] = tuple(self._agent_is_terminal(vehicle) for vehicle in self.controlled_vehicles)
        info["agent_arrived"] = tuple(self.has_arrived(vehicle) for vehicle in self.controlled_vehicles)
        return info

    def _reset(self) -> None:
        self.object_index = 0
        self._make_road()
        self._make_vehicles(self.config["initial_vehicle_count"])

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        obs, reward, done, info = super().step(action)
        self._clear_vehicles()
        self._spawn_vehicle(spawn_probability=self.config["spawn_probability"])
        return obs, reward, done, info

    def _make_road(self) -> None:
        """
        Make an 4-way intersection.

        The horizontal road has the right of way. More precisely, the levels of priority are:
            - 3 for horizontal straight lanes and right-turns
            - 1 for vertical straight lanes and right-turns
            - 2 for horizontal left-turns
            - 0 for vertical left-turns

        The code for nodes in the road network is:
        (o:outer | i:inner + [r:right, l:left]) + (0:south | 1:west | 2:north | 3:east)

        :return: the intersection road
        """
        lane_width = 5 #AbstractLane.DEFAULT_WIDTH
        right_turn_radius = lane_width + 5  # [m}
        left_turn_radius = right_turn_radius + lane_width  # [m}
        outer_distance = right_turn_radius  + lane_width / 2
        access_length = 25  # [m]

        net = RoadNetwork()
        n, c, s = LineType.NONE, LineType.CONTINUOUS, LineType.STRIPED
        for corner in range(4):
            angle = np.radians(90 * corner)
            is_horizontal = corner % 2
            priority = 3 if is_horizontal else 1
            rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
            # Incoming
            start = rotation @ np.array([lane_width / 2, access_length + outer_distance])
            end = rotation @ np.array([lane_width / 2, outer_distance])
            net.add_lane("o" + str(corner), "ir" + str(corner),
                         StraightLane(start, end, line_types=[s, c], priority=priority, speed_limit=10))
            # Right turn
            r_center = rotation @ (np.array([outer_distance, outer_distance]))
            net.add_lane("ir" + str(corner), "il" + str((corner - 1) % 4),
                         CircularLane(r_center, right_turn_radius, angle + np.radians(180), angle + np.radians(270),
                                      line_types=[n, c], priority=priority, speed_limit=10))
            # Left turn
            l_center = rotation @ (np.array([-left_turn_radius + lane_width / 2, left_turn_radius - lane_width / 2]))
            net.add_lane("ir" + str(corner), "il" + str((corner + 1) % 4),
                         CircularLane(l_center, left_turn_radius, angle + np.radians(0), angle + np.radians(-90),
                                      clockwise=False, line_types=[n, n], priority=priority - 1, speed_limit=10))
            # Straight
            start = rotation @ np.array([lane_width / 2, outer_distance])
            end = rotation @ np.array([lane_width / 2, -outer_distance])
            net.add_lane("ir" + str(corner), "il" + str((corner + 2) % 4),
                         StraightLane(start, end, line_types=[n, n], priority=priority, speed_limit=10))
            # Exit
            start = rotation @ np.flip([lane_width / 2, access_length + outer_distance], axis=0)
            end = rotation @ np.flip([lane_width / 2, outer_distance], axis=0)
            net.add_lane("il" + str((corner - 1) % 4), "o" + str((corner - 1) % 4),
                         StraightLane(end, start, line_types=[n, c], priority=priority, speed_limit=10))

        road = RegulatedRoad(network=net, np_random=self.np_random, record_history=self.config["show_trajectories"])
        self.road = road

    def _make_vehicles(self, n_vehicles: int = 10) -> None:
        """
        Populate a road with several vehicles on the highway and on the merging lane

        :return: the ego-vehicle
        """
        # Configure vehicles
        vehicle_type = utils.class_from_path(self.config["other_vehicles_type"])
        vehicle_type.DISTANCE_WANTED = 7  # Low jam distance
        vehicle_type.COMFORT_ACC_MAX = 6
        vehicle_type.COMFORT_ACC_MIN = -3

        # Random vehicles
        simulation_steps = 1
        for t in range(n_vehicles - 1):
            self._spawn_vehicle(np.linspace(0, 80, n_vehicles)[t])
        for _ in range(simulation_steps):
            [(self.road.act(), self.road.step(1 / self.config["simulation_frequency"])) for _ in range(self.config["simulation_frequency"])]

        # Challenger vehicle
        self._spawn_vehicle(10, spawn_probability=0.3, go_straight=True, position_deviation=0.1, speed_deviation=0)

        # Controlled vehicles
        self.controlled_vehicles = []
        for ego_id in range(0, self.config["controlled_vehicles"]):
            ego_lane = self.road.network.get_lane(("o{}".format(ego_id % 4), "ir{}".format(ego_id % 4), 0))
            # static destination or random destination
            destination = self.config["destination"] or "o" + str(self.np_random.randint(1, 4))
            ego_vehicle = self.action_type.vehicle_class(
                             self.road,
                             ego_lane.position(5*self.np_random.randn(1), 0),
                             speed=ego_lane.speed_limit,
                             heading=ego_lane.heading_at(0))
            self.object_index += 1
            ego_vehicle.index = self.object_index
            try:
                ego_vehicle.plan_route_to(destination)
                ego_vehicle.SPEED_MIN = 0
                ego_vehicle.SPEED_MAX = self.config["reward_speed_range"][1]
                ego_vehicle.SPEED_COUNT = 3
                ego_vehicle.speed_index = ego_vehicle.speed_to_index(ego_lane.speed_limit)
                ego_vehicle.target_speed = ego_vehicle.index_to_speed(ego_vehicle.speed_index)
            except AttributeError:
                pass

            self.road.vehicles.append(ego_vehicle)
            self.controlled_vehicles.append(ego_vehicle)
            for v in self.road.vehicles:  # Prevent early collisions
                if v is not ego_vehicle and np.linalg.norm(v.position - ego_vehicle.position) < 20:
                    self.road.vehicles.remove(v)

    def _spawn_vehicle(self,
                       longitudinal: float = 0,
                       position_deviation: float = 1.,
                       speed_deviation: float = 1.,
                       spawn_probability: float = 0.6,
                       go_straight: bool = False) -> None:
        if self.np_random.rand() > spawn_probability:
            return
        if len(self.road.vehicles) > self.config["max_vehicle_count"]:
            return

        self.object_index += 1
        route = self.np_random.choice(range(4), size=2, replace=False)
        route[1] = (route[0] + 2) % 4 if go_straight else route[1]
        vehicle_type = utils.class_from_path(self.config["other_vehicles_type"])
        vehicle = vehicle_type.make_on_lane(self.road, ("o" + str(route[0]), "ir" + str(route[0]), 0),
                                            longitudinal=longitudinal + 5 + self.np_random.randn() * position_deviation,
                                            speed=8 + self.np_random.randn() * speed_deviation)
        vehicle.index = self.object_index
        for v in self.road.vehicles:
            if np.linalg.norm(v.position - vehicle.position) < 15:
                return
        vehicle.plan_route_to("o" + str(route[1]))
        vehicle.randomize_behavior()
        self.road.vehicles.append(vehicle)
        return vehicle

    def _clear_vehicles(self) -> None:
        is_leaving = lambda vehicle: "il" in vehicle.lane_index[0] and "o" in vehicle.lane_index[1] \
                                     and vehicle.lane.local_coordinates(vehicle.position)[0] \
                                     >= vehicle.lane.length - 4 * vehicle.LENGTH
        self.road.vehicles = [vehicle for vehicle in self.road.vehicles if
                              vehicle in self.controlled_vehicles or not (is_leaving(vehicle) or vehicle.route is None)]

    def has_arrived(self, vehicle: Vehicle, exit_distance: float = 25) -> bool:
        exit_distance = vehicle.lane.length - vehicle.LENGTH
        return "il" in vehicle.lane_index[0] \
               and "o" in vehicle.lane_index[1] \
               and vehicle.lane.local_coordinates(vehicle.position)[0] >= exit_distance

    def _cost(self, action: int) -> float:
        """The constraint signal is the occurrence of collisions."""
        return float(self.vehicle.crashed)


class IntersectionNoStopEnv(IntersectionNewEnv):
    '''When crashes, don't stop the environment, just give the reward!'''

    def _is_terminal(self) -> bool:
        return all(self.has_arrived(vehicle) for vehicle in self.controlled_vehicles) \
               or self.steps >= self.config["duration"] * self.config["policy_frequency"]

    def _agent_is_terminal(self, vehicle: Vehicle) -> bool:
        """The episode is over when a collision occurs or when the access ramp has been passed."""
        return self.steps >= self.config["duration"] * self.config["policy_frequency"] \
            or self.has_arrived(vehicle)

    def _info(self, obs: np.ndarray, action: int) -> dict:
        info = super()._info(obs, action)
        info["agents_rewards"] = tuple(self._agent_reward(action, vehicle) for vehicle in self.controlled_vehicles)
        info["agents_dones"] = tuple(self._agent_is_terminal(vehicle) for vehicle in self.controlled_vehicles)
        info["agent_arrived"] = tuple(self.has_arrived(vehicle) for vehicle in self.controlled_vehicles)
        info["agents_crashed"] = tuple(v.CRASHED_FLAG for v in self.controlled_vehicles)
        return info

    def _agent_reward(self, action: int, vehicle: Vehicle) -> float:
        scaled_speed = utils.lmap(self.vehicle.speed, self.config["reward_speed_range"], [0, 1])


        # if vehicle.CURRENT_CRASH:
        # crash_ratio = 1 if vehicle.CURRENT_DIST < vehicle.WIDTH+1 else -0.5 * vehicle.CURRENT_DIST + 2.5
        crash_ratio = 1

        # reward = self.config["collision_reward"] * vehicle.CURRENT_CRASH * crash_ratio\
        #          + self.config["high_speed_reward"] * np.clip(scaled_speed, 0, 1)
        if self.config["no_speed_reward"]:
            reward = self.config["collision_reward"] * vehicle.CURRENT_CRASH * crash_ratio + self.config["step_reward"]
        else:
            reward = self.config["collision_reward"] * vehicle.CURRENT_CRASH * crash_ratio \
                         + self.config["high_speed_reward"] * np.clip(scaled_speed, 0, 1)

        reward = self.config["arrived_reward"] if self.has_arrived(vehicle) else reward
        vehicle.CURRENT_CRASH = False
        if self.config["normalize_reward"]:
            reward = utils.lmap(reward, [self.config["collision_reward"], self.config["arrived_reward"]], [0, 1])
        return reward

    def _make_vehicles(self, n_vehicles: int = 10) -> None:
        """
        Populate a road with several vehicles on the highway and on the merging lane

        :return: the ego-vehicle
        """
        # Configure vehicles
        vehicle_type = utils.class_from_path(self.config["other_vehicles_type"])
        vehicle_type.DISTANCE_WANTED = 7  # Low jam distance
        vehicle_type.COMFORT_ACC_MAX = 6
        vehicle_type.COMFORT_ACC_MIN = -3

        # Random vehicles
        simulation_steps = 1
        for t in range(n_vehicles - 1):
            self._spawn_vehicle(np.linspace(0, 80, n_vehicles)[t])
        for _ in range(simulation_steps):
            [(self.road.act(), self.road.step(1 / self.config["simulation_frequency"])) for _ in range(self.config["simulation_frequency"])]

        # Challenger vehicle
        self._spawn_vehicle(10, spawn_probability=0.3, go_straight=True, position_deviation=0.1, speed_deviation=0)

        # Controlled vehicles
        self.controlled_vehicles = []
        for ego_id in range(0, self.config["controlled_vehicles"]):
            ego_lane = self.road.network.get_lane(("o{}".format(ego_id % 4), "ir{}".format(ego_id % 4), 0))
            # static destination or random destination
            destination = self.config["destination"] or "o" + str(self.np_random.randint(1, 4))
            # from highway_env.vehicle.controller import MDPNoColVehicle
            ego_vehicle = MDPNoColVehicle(
                             self.road,
                             ego_lane.position(5*self.np_random.randn(1), 0),
                             speed=ego_lane.speed_limit,
                             heading=ego_lane.heading_at(0))
            self.object_index += 1
            ego_vehicle.index = self.object_index
            try:
                ego_vehicle.plan_route_to(destination)
                ego_vehicle.SPEED_MIN = 0
                ego_vehicle.SPEED_MAX = self.config["reward_speed_range"][1]
                ego_vehicle.SPEED_COUNT = 3
                ego_vehicle.speed_index = ego_vehicle.speed_to_index(ego_lane.speed_limit)
                ego_vehicle.target_speed = ego_vehicle.index_to_speed(ego_vehicle.speed_index)
            except AttributeError:
                pass

            self.road.vehicles.append(ego_vehicle)
            self.controlled_vehicles.append(ego_vehicle)
            for v in self.road.vehicles:  # Prevent early collisions
                if v is not ego_vehicle and np.linalg.norm(v.position - ego_vehicle.position) < 20:
                    self.road.vehicles.remove(v)
    def _spawn_vehicle(self,
                       longitudinal: float = 0,
                       position_deviation: float = 1.,
                       speed_deviation: float = 1.,
                       spawn_probability: float = 0.6,
                       go_straight: bool = False) -> None:
        if self.np_random.rand() > spawn_probability:
            return
        if len(self.road.vehicles) > self.config["max_vehicle_count"]:
            return

        self.object_index += 1
        route = self.np_random.choice(range(4), size=2, replace=False)
        route[1] = (route[0] + 2) % 4 if go_straight else route[1]
        vehicle_type = IDMNoColVehicle
        vehicle = vehicle_type.make_on_lane(self.road, ("o" + str(route[0]), "ir" + str(route[0]), 0),
                                            longitudinal=longitudinal + 5 + self.np_random.randn() * position_deviation,
                                            speed=8 + self.np_random.randn() * speed_deviation)
        vehicle.index = self.object_index
        for v in self.road.vehicles:
            if np.linalg.norm(v.position - vehicle.position) < 15:
                return
        vehicle.plan_route_to("o" + str(route[1]))
        vehicle.randomize_behavior()
        self.road.vehicles.append(vehicle)
        return vehicle

class IntersectionDisPunish(IntersectionNewEnv):
    '''Punish if the vehicle is too close to other vehicles'''

    def _agent_reward(self, action: int, vehicle: Vehicle) -> float:
        scaled_speed = utils.lmap(self.vehicle.speed, self.config["reward_speed_range"], [0, 1])

        crash_reward = self.config["collision_reward"] * vehicle.crashed

        if not vehicle.crashed:
            for v in self.road.vehicles:
                if v != vehicle:
                    dis = np.linalg.norm(v.position - vehicle.position)
                    direction = np.arctan2(v.position[1]-vehicle.position[1], v.position[0]-vehicle.position[0])
                    angle = utils.wrap_to_pi(direction - vehicle.heading)
                    sign = 1 if abs(angle) < np.pi/2 else 0
                    dis_longi, dis_lateral = abs(dis * np.cos(angle)), abs(dis * np.sin(angle))
                    if dis_longi <= vehicle.LENGTH*2 + vehicle.speed * 1. / self.config['policy_frequency'] and \
                        dis_lateral <= vehicle.WIDTH*2:
                       crash_reward += vehicle.LENGTH/dis * self.config["collision_reward"] * sign

        # if vehicle.impact is not None and not vehicle.crashed:
        #     crash_reward = self.config["collision_reward"] * 0.5
        # elif vehicle.crashed:
        #     crash_reward = self.config["collision_reward"]
        # reward = self.config["collision_reward"] * vehicle.crashed \
        #          + self.config["high_speed_reward"] * np.clip(scaled_speed, 0, 1)
        reward = crash_reward + self.config["high_speed_reward"] * np.clip(scaled_speed, 0, 1)
        # reward = self.config["collision_reward"] * vehicle.crashed + highspeed_reward

        reward = self.config["arrived_reward"] if self.has_arrived(vehicle) else reward
        if self.config["normalize_reward"]:
            reward = utils.lmap(reward, [self.config["collision_reward"], self.config["arrived_reward"]], [0, 1])
        return reward

class IntersectionBaselineEnv(IntersectionNewEnv):
    def _make_vehicles(self, n_vehicles: int = 10) -> None:
        """
        Populate a road with several vehicles on the highway and on the merging lane

        :return: the ego-vehicle
        """
        # Configure vehicles
        vehicle_type = utils.class_from_path(self.config["other_vehicles_type"])
        vehicle_type.DISTANCE_WANTED = 7  # Low jam distance
        vehicle_type.COMFORT_ACC_MAX = 6
        vehicle_type.COMFORT_ACC_MIN = -3

        # Random vehicles
        simulation_steps = 1
        for t in range(n_vehicles - 1):
            self._spawn_vehicle(np.linspace(0, 80, n_vehicles)[t])
        for _ in range(simulation_steps):
            [(self.road.act(), self.road.step(1 / self.config["simulation_frequency"])) for _ in range(self.config["simulation_frequency"])]

        # Challenger vehicle
        self._spawn_vehicle(10, spawn_probability=0.3, go_straight=True, position_deviation=0.1, speed_deviation=0)

        # Controlled vehicles
        self.controlled_vehicles = []
        for ego_id in range(0, self.config["controlled_vehicles"]):
            ego_lane = self.road.network.get_lane(("o{}".format(ego_id % 4), "ir{}".format(ego_id % 4), 0))
            # static destination or random destination
            destination = self.config["destination"] or "o" + str(self.np_random.randint(1, 4))
            ego_vehicle_type = IDMBaseLineVehicle
            ego_vehicle = ego_vehicle_type(
                             self.road,
                             ego_lane.position(5*self.np_random.randn(1), 0),
                             speed=ego_lane.speed_limit,
                             heading=ego_lane.heading_at(0))
            self.object_index += 1
            ego_vehicle.index = self.object_index
            try:
                ego_vehicle.plan_route_to(destination)
                ego_vehicle.SPEED_MIN = 0
                ego_vehicle.SPEED_MAX = self.config["reward_speed_range"][1]
                ego_vehicle.SPEED_COUNT = 3
                ego_vehicle.speed_index = ego_vehicle.speed_to_index(ego_lane.speed_limit)
                ego_vehicle.target_speed = ego_vehicle.index_to_speed(ego_vehicle.speed_index)
            except AttributeError:
                pass

            self.road.vehicles.append(ego_vehicle)
            self.controlled_vehicles.append(ego_vehicle)
            for v in self.road.vehicles:  # Prevent early collisions
                if v is not ego_vehicle and np.linalg.norm(v.position - ego_vehicle.position) < 20:
                    self.road.vehicles.remove(v)


register(
    id='intersection-v1',
    entry_point='highway_env.envs:IntersectionNewEnv',
)

register(
    id='intersection-v2',
    entry_point='highway_env.envs:IntersectionDisPunish',
)

register(
    id='intersection-v3',
    entry_point='highway_env.envs:IntersectionNoStopEnv',
)

register(
    id='intersection-v4',
    entry_point='highway_env.envs:IntersectionBaselineEnv',
)


