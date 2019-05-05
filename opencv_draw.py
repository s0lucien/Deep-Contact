'''
In order to visiulize the world when it is running, make more sense.
'''
import cv2
import numpy as np

from Box2D import (b2Color, b2DistanceJoint, b2MouseJoint, b2PulleyJoint)
from Box2D.b2 import (staticBody, dynamicBody, polygonShape, circleShape, loopShape)


def cvcolor(color):
    return int(255.0 * color[2]), int(255.0 * color[1]), int(255.0 * color[0])


def cvcoord(pos):
    return tuple(map(int, pos))


class OpencvDrawFuncs(object):
    def __init__(self, w, h, ppm, fill_polygon=True, flip_y=True):
        self._w = w
        self._h = h
        self._ppm = ppm
        self._colors = {
            staticBody: (255, 255, 255),
            dynamicBody: (127, 127, 127),
        }
        self._fill_polygon = fill_polygon
        self._flip_y = flip_y
        self.screen = np.zeros((self._h, self._w, 3), np.uint8)

    def install(self):
        polygonShape.draw = self._draw_polygon
        circleShape.draw = self._draw_circle
        loopShape.draw = self._draw_loop

    def draw_world(self, world):
        for body in world.bodies:
            for fixture in body.fixtures:
                fixture.shape.draw(body, fixture)

    def clear_screen(self, screen=None):
        if screen is None:
            self.screen.fill(0)
        else:
            self.screen = screen

    def _fix_vertices(self, vertices):
        if self._flip_y:
            return [(v[0], self._h - v[1]) for v in vertices]
        else:
            return [(v[0], v[1]) for v in vertices]

    def _draw_polygon(self, body, fixture):
        polygon = fixture.shape

        transform = body.transform
        vertices = self._fix_vertices([transform * v * self._ppm
                                       for v in polygon.vertices])

        pts = np.array(vertices, np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(self.screen, [pts], True, self._colors[body.type])

        if self._fill_polygon:
            lightc = np.array(self._colors[body.type], dtype=int) * 0.5
            cv2.fillPoly(self.screen, [pts], lightc)

    def _draw_circle(self, body, fixture):
        circle = fixture.shape
        position = self._fix_vertices(
            [body.transform * circle.pos * self._ppm])[0]
        cv2.circle(self.screen, cvcoord(position), int(
            circle.radius * self._ppm), self._colors[body.type], -1)

    def _draw_loop(self, body, fixture):
        loop = fixture.shape
        transform = body.transform
        vertices = self._fix_vertices([transform * v * self._ppm
                                       for v in loop.vertices])
        v1 = vertices[-1]
        for v2 in vertices:
            cv2.line(self.screen, cvcoord(v1), cvcoord(v2),
                     self._colors[body.type], 1)
            v1 = v2
