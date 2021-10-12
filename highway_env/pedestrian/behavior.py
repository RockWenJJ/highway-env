from typing import Tuple, Union

import numpy as np
import math
import copy
from highway_env.road.road import Road, Route, LaneIndex
from highway_env.utils import Vector
from highway_env import utils
from highway_env.pedestrian.kinematics import Pedestrian
from highway_env.vehicle.controller import MDPVehicle


class GeneralPedestrian(Pedestrian):
    """
    A vehicle using both a longitudinal and a lateral decision policies.

    - Longitudinal: the IDM model computes an acceleration given the preceding vehicle's distance and speed.
    - Lateral: the MOBIL model decides when to change lane by maximizing the acceleration of nearby vehicles.
    """

    # Longitudinal policy parameters
    ACC_MAX = 1.5  # [m/s2]
    """Maximum acceleration."""

    COMFORT_ACC_MAX = 0.5  # [m/s2]
    """Desired maximum acceleration."""

    COMFORT_ACC_MIN = -0.5  # [m/s2]
    """Desired maximum deceleration."""

    DISTANCE_WANTED = 0.5 + Pedestrian.RADIUS*2  # [m]
    """Desired jam distance to the front pedestrian."""

    TIME_WANTED = 0.5  # [s]
    """Desired time gap to the front pedestrian."""

    DELTA = 4.0  # []
    """Exponent of the velocity term."""

    DELTA_RANGE = [3.5, 4.5]
    """Range of delta when chosen randomly."""

    # Lateral policy parameters
    POLITENESS = 0.  # in [0, 1]
    LANE_CHANGE_MIN_ACC_GAIN = 0.2  # [m/s2]
    LANE_CHANGE_MAX_BRAKING_IMPOSED = 2.0  # [m/s2]
    LANE_CHANGE_DELAY = 1.0  # [s]

    def __init__(self,
                 road: Road,
                 position: Vector,
                 target: Vector,
                 heading: float = 0,
                 speed: float = 0,
                 target_lane_index: int = None,
                 target_speed: float = None,
                 enable_lane_change: bool = True,
                 timer: float = None):
        super().__init__(road, position, target, heading, speed, target)
        self.target_lane_index = self.lane_index
        self.enable_lane_change = enable_lane_change
        self.timer = timer or (np.sum(self.position)*np.pi) % self.LANE_CHANGE_DELAY

    def randomize_behavior(self):
        self.DELTA = self.road.np_random.uniform(low=self.DELTA_RANGE[0], high=self.DELTA_RANGE[1])

    @classmethod
    def create_from(cls, pedestrian: Pedestrian) -> "GeneralPedestrian":
        """
        Create a new vehicle from an existing one.

        The vehicle dynamics and target dynamics are copied, other properties are default.

        :param vehicle: a vehicle
        :return: a new vehicle at the same dynamical state
        """
        v = cls(pedestrian.road, pedestrian.position, heading=pedestrian.heading, speed=pedestrian.speed,
                target_lane_index=pedestrian.target_lane_index, target_speed=pedestrian.target_speed,
                route=pedestrian.route, timer=getattr(pedestrian, 'timer', None))
        return v

    def act(self, action: Union[dict, str] = None):
        """
        Execute an action.

        For now, no action is supported because the vehicle takes all decisions
        of acceleration and lane changes on its own, based on the IDM and MOBIL models.

        :param action: the action
        """
        if self.crashed:
            return
        action = {}

        # Longitudinal: IDM
        vehicles = copy.deepcopy(self.road.vehicles)
        front_vehicle, rear_vehicle = self.neighbour_vehicles(vehicles)

        action['acceleration'] = self.acceleration(ego_vehicle=self,
                                                   front_vehicle=front_vehicle,
                                                   rear_vehicle=rear_vehicle)

        action['acceleration'] = np.clip(action['acceleration'], 0, self.ACC_MAX)
        action['steering'] = 0.0
        Pedestrian.act(self, action)  # Skip ControlledVehicle.act(), or the command will be overriden.

    def step(self, dt: float):
        """
        Step the simulation.

        Increases a timer used for decision policies, and step the vehicle dynamics.

        :param dt: timestep
        """
        self.timer += dt
        super().step(dt)

    def neighbour_vehicles(self, vehicles):
        f_v, r_v = None, None
        if not hasattr(vehicles, '__iter__'):
            return f_v, r_v

        for vehicle in vehicles:
            # ignore controller vehicles
            if isinstance(vehicle, MDPVehicle):
                continue
            pos = vehicle.position
            d = np.linalg.norm(pos-self.position)
            if d < 40:
                angle = math.atan2(pos[1]-self.position[1], pos[0]-self.position[0])
                if abs(angle+self.heading) < math.pi/6 or abs(angle-self.heading) < math.pi/6:
                    f_v = vehicle
                    return f_v, r_v

        return f_v, r_v


    def acceleration(self,
                     ego_vehicle: Pedestrian,
                     front_vehicle: Pedestrian = None,
                     rear_vehicle: Pedestrian = None) -> float:
        """
        Compute an acceleration command with the Intelligent Driver Model.

        The acceleration is chosen so as to:
        - reach a target speed;
        - maintain a minimum safety distance (and safety time) w.r.t the front vehicle.

        :param ego_vehicle: the vehicle whose desired acceleration is to be computed. It does not have to be an
                            IDM vehicle, which is why this method is a class method. This allows an IDM vehicle to
                            reason about other vehicles behaviors even though they may not IDMs.
        :param front_vehicle: the vehicle preceding the ego-vehicle
        :param rear_vehicle: the vehicle following the ego-vehicle
        :return: the acceleration command for the ego-vehicle [m/s2]
        """
        if not ego_vehicle or not isinstance(ego_vehicle, Pedestrian):
            return 0
        ego_target_speed = abs(utils.not_zero(getattr(ego_vehicle, "target_speed", 0)))
        # acceleration = self.COMFORT_ACC_MAX * (
        #         1 - np.power(max(ego_vehicle.speed, 0) / ego_target_speed, self.DELTA))
        acceleration = 0.5

        if front_vehicle:
            d = ego_vehicle.lane_distance_to(front_vehicle)
            acceleration -= self.COMFORT_ACC_MAX * \
                np.power(self.desired_gap(ego_vehicle, front_vehicle) / utils.not_zero(d), 2)
        return acceleration

    def desired_gap(self, ego_vehicle: Pedestrian, front_vehicle: Pedestrian = None, projected: bool = True) -> float:
        """
        Compute the desired distance between a vehicle and its leading vehicle.

        :param ego_vehicle: the vehicle being controlled
        :param front_vehicle: its leading vehicle
        :param projected: project 2D velocities in 1D space
        :return: the desired distance between the two [m]
        """
        d0 = self.DISTANCE_WANTED
        tau = self.TIME_WANTED
        ab = -self.COMFORT_ACC_MAX * self.COMFORT_ACC_MIN
        dv = np.dot(ego_vehicle.velocity - front_vehicle.velocity, ego_vehicle.direction) if projected \
            else ego_vehicle.speed - front_vehicle.speed
        d_star = d0 + ego_vehicle.speed * tau + ego_vehicle.speed * dv / (2 * np.sqrt(ab))
        return d_star

    def change_lane_policy(self) -> None:
        """
        Decide when to change lane.

        Based on:
        - frequency;
        - closeness of the target lane;
        - MOBIL model.
        """
        # If a lane change already ongoing
        if self.lane_index != self.target_lane_index:
            # If we are on correct route but bad lane: abort it if someone else is already changing into the same lane
            if self.lane_index[:2] == self.target_lane_index[:2]:
                for v in self.road.vehicles:
                    if v is not self \
                            and v.lane_index != self.target_lane_index \
                            and isinstance(v, Pedestrian) \
                            and v.target_lane_index == self.target_lane_index:
                        d = self.lane_distance_to(v)
                        d_star = self.desired_gap(self, v)
                        if 0 < d < d_star:
                            self.target_lane_index = self.lane_index
                            break
            return

        # else, at a given frequency,
        if not utils.do_every(self.LANE_CHANGE_DELAY, self.timer):
            return
        self.timer = 0

        # decide to make a lane change
        for lane_index in self.road.network.side_lanes(self.lane_index):
            # Is the candidate lane close enough?
            if not self.road.network.get_lane(lane_index).is_reachable_from(self.position):
                continue
            # Does the MOBIL model recommend a lane change?
            if self.mobil(lane_index):
                self.target_lane_index = lane_index

    def mobil(self, lane_index: LaneIndex) -> bool:
        """
        MOBIL lane change model: Minimizing Overall Braking Induced by a Lane change

            The vehicle should change lane only if:
            - after changing it (and/or following vehicles) can accelerate more;
            - it doesn't impose an unsafe braking on its new following vehicle.

        :param lane_index: the candidate lane for the change
        :return: whether the lane change should be performed
        """
        # Is the maneuver unsafe for the new following vehicle?
        new_preceding, new_following = self.road.neighbour_vehicles(self, lane_index)
        new_following_a = self.acceleration(ego_vehicle=new_following, front_vehicle=new_preceding)
        new_following_pred_a = self.acceleration(ego_vehicle=new_following, front_vehicle=self)
        if new_following_pred_a < -self.LANE_CHANGE_MAX_BRAKING_IMPOSED:
            return False

        # Do I have a planned route for a specific lane which is safe for me to access?
        old_preceding, old_following = self.road.neighbour_vehicles(self)
        self_pred_a = self.acceleration(ego_vehicle=self, front_vehicle=new_preceding)
        if self.route and self.route[0][2]:
            # Wrong direction
            if np.sign(lane_index[2] - self.target_lane_index[2]) != np.sign(self.route[0][2] - self.target_lane_index[2]):
                return False
            # Unsafe braking required
            elif self_pred_a < -self.LANE_CHANGE_MAX_BRAKING_IMPOSED:
                return False

        # Is there an acceleration advantage for me and/or my followers to change lane?
        else:
            self_a = self.acceleration(ego_vehicle=self, front_vehicle=old_preceding)
            old_following_a = self.acceleration(ego_vehicle=old_following, front_vehicle=self)
            old_following_pred_a = self.acceleration(ego_vehicle=old_following, front_vehicle=old_preceding)
            jerk = self_pred_a - self_a + self.POLITENESS * (new_following_pred_a - new_following_a
                                                             + old_following_pred_a - old_following_a)
            if jerk < self.LANE_CHANGE_MIN_ACC_GAIN:
                return False

        # All clear, let's go!
        return True

    # def recover_from_stop(self, acceleration: float) -> float:
    #     """
    #     If stopped on the wrong lane, try a reversing maneuver.
    #
    #     :param acceleration: desired acceleration from IDM
    #     :return: suggested acceleration to recover from being stuck
    #     """
    #     stopped_speed = 1.5
    #     safe_distance = 200
    #     # Is the vehicle stopped on the wrong lane?
    #     if self.speed < stopped_speed:
    #         # Check for free room behind on both lanes
    #         if (not rear or rear.lane_distance_to(self) > safe_distance) and \
    #                 (not new_rear or new_rear.lane_distance_to(self) > safe_distance):
    #             # Reverse
    #             return -self.COMFORT_ACC_MAX / 2
    #     return acceleration
