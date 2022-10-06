"""Time how much slower list comprehension is when accessing attrib of an object of a class vs tuple"""
import numpy as np
import random
from typing import Tuple


class Unit:
    def __init__(self, player: int, pos: Tuple):
        """The unit of each player. New unit spawns every N days. Can be moved in any direction by 1 km"""
        assert 0 <= player < 4

        self.player = int(player)
        self.pos = pos
        self.status = 1  # 1 = alive, 0 = dead


def create_units_class():
    units = []
    # Scatter more throughout the map
    for idx in range(4):
        # for idy in range(100000):
        #     units.append(Unit(idx, (random.random() * 100.0, random.random() * 100.0)))

        units.extend([Unit(idx, (random.random() * 100.0, random.random() * 100.0)) for _ in range(100000)])
    return units


def create_units_tuple():
    units = []
    # Scatter more throughout the map
    for idx in range(4):
        # for idy in range(100000):
        #     units.append((idx, (random.random() * 100.0, random.random() * 100.0)))

        units.extend([(idx, (random.random() * 100.0, random.random() * 100.0)) for _ in range(100000)])
    return units


def list_from_class(units):
    pts = []
    player_idx = []
    for u in units:
        # VECTORIZE: Hash map to detect disputed cells. List of keys/vals, index all cells simultaneously
        pos_int = (int(u.pos[1]), int(u.pos[0]))
        pts.append(pos_int)
        player_idx.append(u.player)

    return pts, player_idx


def list_from_tuple(units):
    pts = []
    player_idx = []
    for pl, (y, x) in units:
        # VECTORIZE: Hash map to detect disputed cells. List of keys/vals, index all cells simultaneously
        pos_int = (int(y), int(x))
        pts.append(pos_int)
        player_idx.append(pl)

    return pts, player_idx


def list_from_tuple_with_hash(units):
    pts_hash = {}
    for pl, (y, x) in units:
        # VECTORIZE: Hash map to detect disputed cells. List of keys/vals, index all cells simultaneously
        pos_int = (int(y), int(x))
        if pos_int in pts_hash:
            player_existing = pts_hash[pos_int]
            if player_existing != pl:
                pts_hash[pos_int] = 4  # Don't count point
        else:
            pts_hash[pos_int] = pl

    pts = pts_hash.keys()
    player_idx = pts_hash.values()
    return pts, player_idx


if __name__ == "__main__":
    units1 = create_units_class()
    units2 = create_units_tuple()

    a, b = list_from_class(units1)
    a, b = list_from_tuple(units2)
    a, b = list_from_tuple_with_hash(units2)

