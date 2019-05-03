"""Iceberg module documentation.

This module creates iceberg objects according to the time, space, velocity, and
geometry specified by the user.

Attributes:
    WATERLINE_LENGTH_RANGE_BY_SIZE (dict): dictionary of iceberg waterline length ranges (min, max)
        in meters for each iceberg size class.
    SAIL_HEIGHT_RANGE_BY_SIZE (dict): dictionary of iceberg sail height ranges (min, max) in meters
        for each iceberg size class.
    HEIGHT_TO_DRAFT_RATIO_BY_SHAPE (dict): dictionary of iceberg sail height to keel depth ratios
        for each iceberg shape class.
    SHAPE_FACTOR_BY_SHAPE (dict): dictionary of iceberg shape factors for each iceberg shape class.

"""

import numpy as np

WATERLINE_LENGTH_RANGE_BY_SIZE = {'GR': (0.01, 5), 'BB': (5, 15), 'SM': (15, 60), 'MED': (60, 120),
                                  'LG': (120, 200), 'VLG': (200, 1000)}
SAIL_HEIGHT_RANGE_BY_SIZE = {'GR': (0.01, 1), 'BB': (1, 5), 'SM': (5, 15), 'MED': (15, 45),
                             'LG': (45, 75), 'VLG': (75, 300)}
HEIGHT_TO_DRAFT_RATIO_BY_SHAPE = {'TAB': 0.2, 'NTAB': 0.2, 'DOM': 0.25, 'PIN': 0.5, 'WDG': 0.2, 'DD': 0.5, 'BLK': 0.2}
SHAPE_FACTOR_BY_SHAPE = {'TAB': 0.5, 'NTAB': 0.41, 'DOM': 0.41, 'PIN': 0.25, 'WDG': 0.33, 'DD': 0.15, 'BLK': 0.5}


class IcebergGeometry:
    """Instantiates object with iceberg geometry according to size and shape class specified.

    Args:
        size (str): iceberg size class as outlined by the IIP.
        shape (str): iceberg shape class as outlined by the IIP.

    Examples:
        >>> iceberg_geometry = IcebergGeometry('LG', 'TAB')
        >>> iceberg_geometry.waterline_length
        160.0

    """

    def __init__(self, size='LG', shape='TAB'):

        if isinstance(size, (tuple, list, np.ndarray)):
            self.custom_waterline_length, self.custom_sail_height = size
            self._size = 'CUSTOM'

        elif isinstance(size, str):
            self._size = size

        else:
            self._size = None

        if isinstance(shape, (tuple, list, np.ndarray)):
            self.custom_height_to_draft_ratio, self.custom_shape_factor = shape
            self._shape = 'CUSTOM'

        elif isinstance(shape, str):
            self._shape = shape

        else:
            self._shape = None

        self._waterline_length = None
        self._sail_height = None
        self._mass = None

    @property
    def waterline_length(self):
        """Return the mean waterline length for the size declared."""
        if self._size == 'CUSTOM':
            self._waterline_length = self.custom_waterline_length
            return self._waterline_length

        elif self._size is None:
            return self._waterline_length

        else:
            self._waterline_length = np.mean(WATERLINE_LENGTH_RANGE_BY_SIZE[self._size])
            return self._waterline_length

    @waterline_length.setter
    def waterline_length(self,  value):
        self._waterline_length = value

    @property
    def top_area(self):
        """Return the area of the top face."""
        return self.waterline_length**2

    @property
    def bottom_area(self):
        """Return the area of the bottom face."""
        return self.waterline_length**2

    @property
    def sail_height(self):
        """Return the mean sail height for the size declared."""
        if self._size == 'CUSTOM':
            self._sail_height = self.custom_sail_height
            return self._sail_height

        elif self._size is None:
            return self._sail_height

        else:
            self._sail_height = np.mean(SAIL_HEIGHT_RANGE_BY_SIZE[self._size])
            return self._sail_height

    @sail_height.setter
    def sail_height(self, value):
        self._sail_height = value

    @property
    def sail_area(self):
        """Return the area of rectangular sail."""
        return self.waterline_length * self.sail_height

    @property
    def height_to_draft_ratio(self):
        """Return the height to draft ratio for the shape declared."""
        if self._shape == 'CUSTOM':
            return self.custom_height_to_draft_ratio
        else:
            return HEIGHT_TO_DRAFT_RATIO_BY_SHAPE[self._shape]

    @property
    def shape_factor(self):
        """Return the shape factor for the shape declared."""
        if self._shape == 'CUSTOM':
            return self.custom_shape_factor
        else:
            return SHAPE_FACTOR_BY_SHAPE[self._shape]

    @property
    def keel_depth(self):
        """Return the keel depth for the shape and sail height declared."""
        h2d_ratio = self.height_to_draft_ratio
        sail_height = self.sail_height
        return sail_height / h2d_ratio

    @property
    def mass(self):
        """Return the mass using formula from Rudkin, 2005."""
        if self._size is None:
            return self._mass
        else:
            factor = self.shape_factor
            length = self.waterline_length
            height = self.sail_height
            self._mass = 7.12e3 * factor * length**2 * height
            return self._mass

    @mass.setter
    def mass(self, value):
        self._mass = value

    @property
    def keel_area(self):
        """Return the rectangular keel area."""
        return self.waterline_length * self.keel_depth


class Iceberg:
    """Creates iceberg object."""

    DENSITY = 900

    FORM_DRAG_COEFFICIENT_IN_AIR = 1.5
    FORM_DRAG_COEFFICIENT_IN_WATER = 1.5
    SKIN_DRAG_COEFFICIENT_IN_AIR = 2.5e-4
    SKIN_DRAG_COEFFICIENT_IN_WATER = 5e-4

    def __init__(self, time, position, velocity, geometry, **kwargs):
        self.time = time
        self.latitude, self.longitude = position
        self.eastward_velocity, self.northward_velocity = velocity
        self.geometry = geometry
        self.name = kwargs.get('name', None)
        self.history = {'time': [], 'latitude': [], 'longitude': [],
                        'eastward_velocity': [], 'northward_velocity': []}

    def update_history(self):
        self.history['time'].append(self.time)
        self.history['latitude'].append(self.latitude)
        self.history['longitude'].append(self.longitude)
        self.history['eastward_velocity'].append(self.eastward_velocity)
        self.history['northward_velocity'].append(self.northward_velocity)

    def reset(self):
        self.time = self.history['time'][0]
        self.history['time'] = []
        self.latitude = self.history['latitude'][0]
        self.history['latitude'] = []
        self.longitude = self.history['longitude'][0]
        self.history['longitude'] = []
        self.eastward_velocity = self.history['eastward_velocity'][0]
        self.history['eastward_velocity'] = []
        self.northward_velocity = self.history['northward_velocity'][0]
        self.history['northward_velocity'] = []


def quickstart(time, position, velocity=(0, 0), size='LG', shape='TAB'):
    """Creates iceberg object from minimal args.

    Args:
        time (numpy.datetime64): time of the iceberg at specified position.
        position (tuple of float): latitude, longitude position of iceberg at specified time.

    Returns:
        An object of the Iceberg class.

    """

    geometry = IcebergGeometry(size=size, shape=shape)
    iceberg = Iceberg(time, position, velocity, geometry)

    return iceberg
