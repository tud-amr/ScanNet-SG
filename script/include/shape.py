from enum import Enum
import numpy as np
from typing import Dict, List, Optional, Union, Tuple
from dataclasses import dataclass

'''
This file contains the definitions for the Shape class and its subclasses.
Two types of shapes are supported for now:
- Cylinder: A cylinder shape with a radius and height
- OrientedBox: An oriented bounding box with a length, width, and height
'''
class ShapeType(Enum):
    CYLINDER = "cylinder"
    ORIENTED_BOX = "oriented_box"

@dataclass
class Orientation:
    """Quaternion representation for rotation"""
    x: float
    y: float
    z: float
    w: float

@dataclass
class Cylinder:
    """Cylinder shape representation"""
    radius: float
    height: float
    orientation: Orientation

    def __post_init__(self):
        if self.radius <= 0:
            raise ValueError("Cylinder radius must be positive")
        if self.height <= 0:
            raise ValueError("Cylinder height must be positive")
    
    @classmethod
    def from_dict(cls, data):
        return cls(data['radius'], data['height'], Orientation(data['orientation']['x'], data['orientation']['y'], data['orientation']['z'], data['orientation']['w']))

@dataclass
class OrientedBox:
    """Oriented Bounding Box representation"""
    length: float  # x-dimension
    width: float   # y-dimension
    height: float  # z-dimension
    orientation: Orientation 

    def __post_init__(self):
        if self.length <= 0 or self.width <= 0 or self.height <= 0:
            raise ValueError("Box dimensions must be positive")

    @classmethod
    def from_dict(cls, data):
        return cls(data['length'], data['width'], data['height'], Orientation(data['orientation']['x'], data['orientation']['y'], data['orientation']['z'], data['orientation']['w']))

