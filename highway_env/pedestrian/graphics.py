import itertools
from typing import List, Tuple, TYPE_CHECKING

import numpy as np
import pygame

from highway_env.utils import Vector
from highway_env.vehicle.dynamics import BicycleVehicle
from highway_env.pedestrian.kinematics import Pedestrian
from highway_env.vehicle.controller import ControlledVehicle, MDPVehicle
from highway_env.vehicle.behavior import IDMVehicle, LinearVehicle

if TYPE_CHECKING:
    from highway_env.road.graphics import WorldSurface


class PedestrianGraphics(object):
    RED = (255, 100, 100)
    GREEN = (50, 200, 0)
    BLUE = (100, 200, 255)
    YELLOW = (200, 200, 0)
    BLACK = (60, 60, 60)
    PURPLE = (200, 0, 150)
    DEFAULT_COLOR = YELLOW
    EGO_COLOR = GREEN

    @classmethod
    def display(cls, pedestrian: Pedestrian, surface: "WorldSurface",
                transparent: bool = False,
                offscreen: bool = False,
                label: bool = False,
                draw_roof: bool = False) -> None:
        """
        Display a pedestrian on a pygame surface.

        The vehicle is represented as a colored rotated rectangle.

        :param vehicle: the vehicle to be drawn
        :param surface: the surface to draw the vehicle on
        :param transparent: whether the vehicle should be drawn slightly transparent
        :param offscreen: whether the rendering should be done offscreen or not
        :param label: whether a text label should be rendered
        """
        # if not surface.is_visible(pedestrian.position):
        #     return

        p = pedestrian
        # Pedstrian cricle
        r = p.RADIUS
        pix_c = surface.pix(r)
        pix_r = surface.pix(r)
        s = pygame.Surface((pix_r*2, pix_r*2),flags=pygame.SRCALPHA)  # per-pixel alpha

        color = cls.get_color(p, transparent)
        position = (surface.pos2pix(p.position[0], p.position[1]))
        pygame.draw.circle(s, color, (pix_c, pix_c), pix_r, 0)
        if not offscreen:  # convert_alpha throws errors in offscreen mode TODO() Explain why
            s = pygame.Surface.convert_alpha(s)
        h = p.heading if abs(p.heading) > 2 * np.pi / 180 else 0
        # Centered rotation
        cls.blit_rotate(surface, s, position, np.rad2deg(-h))

    @staticmethod
    def blit_rotate(surf: pygame.SurfaceType, image: pygame.SurfaceType, pos: Vector, angle: float,
                    origin_pos: Vector = None, show_rect: bool = False) -> None:
        """Many thanks to https://stackoverflow.com/a/54714144."""
        # calculate the axis aligned bounding box of the rotated image
        w, h = image.get_size()
        box = [pygame.math.Vector2(p) for p in [(0, 0), (w, 0), (w, -h), (0, -h)]]
        box_rotate = [p.rotate(angle) for p in box]
        min_box = (min(box_rotate, key=lambda p: p[0])[0], min(box_rotate, key=lambda p: p[1])[1])
        max_box = (max(box_rotate, key=lambda p: p[0])[0], max(box_rotate, key=lambda p: p[1])[1])

        # calculate the translation of the pivot
        if origin_pos is None:
            origin_pos = w / 2, h / 2
        pivot = pygame.math.Vector2(origin_pos[0], -origin_pos[1])
        pivot_rotate = pivot.rotate(angle)
        pivot_move = pivot_rotate - pivot

        # calculate the upper left origin of the rotated image
        origin = (pos[0] - origin_pos[0] + min_box[0] - pivot_move[0], pos[1] - origin_pos[1] - max_box[1] + pivot_move[1])
        # get a rotated image
        rotated_image = pygame.transform.rotate(image, angle)
        # rotate and blit the image
        surf.blit(rotated_image, origin)
        # draw rectangle around the image
        if show_rect:
            pygame.draw.rect(surf, (255, 0, 0), (*origin, *rotated_image.get_size()), 2)

    @classmethod
    def display_trajectory(cls, states: List[Pedestrian], surface: "WorldSurface", offscreen: bool = False) -> None:
        """
        Display the whole trajectory of a vehicle on a pygame surface.

        :param states: the list of vehicle states within the trajectory to be displayed
        :param surface: the surface to draw the vehicle future states on
        :param offscreen: whether the rendering should be done offscreen or not
        """
        for vehicle in states:
            cls.display(vehicle, surface, transparent=True, offscreen=offscreen)

    @classmethod
    def display_history(cls, pedestrian: Pedestrian, surface: "WorldSurface", frequency: float = 3, duration: float = 2,
                        simulation: int = 15, offscreen: bool = False) -> None:
        """
        Display the whole trajectory of a vehicle on a pygame surface.

        :param vehicle: the vehicle states within the trajectory to be displayed
        :param surface: the surface to draw the vehicle future states on
        :param frequency: frequency of displayed positions in history
        :param duration: length of displayed history
        :param simulation: simulation frequency
        :param offscreen: whether the rendering should be done offscreen or not
        """
        for v in itertools.islice(pedestrian.history,
                                  None,
                                  int(simulation * duration),
                                  int(simulation / frequency)):
            cls.display(v, surface, transparent=True, offscreen=offscreen)

    @classmethod
    def get_color(cls, pedestrian: Pedestrian, transparent: bool = False) -> Tuple[int]:
        color = cls.DEFAULT_COLOR
        if getattr(pedestrian, "color", None):
            color = pedestrian.color
        elif pedestrian.crashed:
            color = cls.RED
        else:
            color = cls.YELLOW # default color
        if transparent:
            color = (color[0], color[1], color[2], 30)
        return color

    @classmethod
    def darken(cls, color, ratio=0.83):
        return (
            int(color[0] * ratio),
            int(color[1] * ratio),
            int(color[2] * ratio),
        ) + color[3:]

    @classmethod
    def lighten(cls, color, ratio=0.68):
        return (
            min(int(color[0] / ratio), 255),
            min(int(color[1] / ratio), 255),
            min(int(color[2] / ratio), 255),
        ) + color[3:]
