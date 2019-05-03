import numpy as np
from xarray.core.dataarray import DataArray


class UniformRegularLinearInterpolator:

    def __init__(self, grid_vectors, *data, **kwargs):

        self.data = data
        self.grid_vectors = grid_vectors
        self.reference_time = kwargs.pop('reference_time', np.datetime64('1950-01-01T00:00'))
        self.time_units = kwargs.pop('time_units', 'h')
        self.grid_info = get_grid_info(self.grid_vectors, self.reference_time, self.time_units)

    def interpolate(self, point):

        point_list = []

        for p in point:
            if isinstance(p, np.datetime64):
                p = (p - self.reference_time) / np.timedelta64(1, self.time_units)
            point_list.append(p)

        point = tuple(point_list)

        return linear_interpolation_on_uniform_regular_grid(self.grid_info, point, *self.data)


def linear_interpolation_on_uniform_regular_grid(grid_info, point, *data):

    data_list = [None] * len(data)

    for dim in range(len(point)):

        x0, dx, xn = grid_info[dim]
        xi = point[dim]

        try:
            assert x0 <= xi <= xn, f'Point out of range in dim {dim} ({xi} is not in ({x0}, {xn})).'
        except TypeError as e:
            print(e, f'(in dim {dim}: x0 = {x0}, xi = {xi}, and xn = {xn})')

        index = (xi - x0) / dx
        index_floor = int(np.floor(index))
        index_diff = index - index_floor

        for i, data_ in enumerate(data):
            data_slice = data_[index_floor: index_floor + 2, ...]
            data_list[i] = (1 - index_diff) * data_slice[0, ...] + index_diff * data_slice[1, ...]

        data = np.array(data_list)

    if len(data) == 1:
        return data[0]

    else:
        return data


def get_grid_info(grid_vectors, reference_time=np.datetime64('1950-01-01T00:00'), time_units='h'):

    grid_info = []

    for grid_vector in grid_vectors:

        if isinstance(grid_vector, DataArray):
            grid_vector = grid_vector.values

        if isinstance(grid_vector[0], np.datetime64):
            grid_vector = (grid_vector - reference_time) / np.timedelta64(1, time_units)

        grid_vector_min = np.min(grid_vector[0])
        grid_vector_max = np.max(grid_vector[-1])
        grid_vector_step = np.mean(np.diff(grid_vector))
        grid_info.append([grid_vector_min, grid_vector_step, grid_vector_max])

    return grid_info
