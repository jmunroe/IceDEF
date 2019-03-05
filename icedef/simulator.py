"""This module sets up and runs iceberg drift simulations and optimizations.
"""

import numpy as np
import xarray as xr
from copy import deepcopy
from scipy.optimize import minimize
from icedef import iceberg, metocean, drift, tools, timesteppers, plot


from logging import getLogger, FileHandler, DEBUG, Formatter
from time import gmtime, strftime


class DebugFileHandler(FileHandler):
    """This class handles the debugging of iceberg simulations by writing the forces to file.
    """
    def __init__(self, filename='debug.log', mode='w', encoding=None, delay=False):
        FileHandler.__init__(self, filename, mode, encoding, delay)
        self.formatter = Formatter('%(message)s')
        self.setFormatter(self.formatter)


class Results:

    def __init__(self, **kwargs):

        self.map = kwargs.pop('map', plot.get_map())
        self.data = {}

    def add(self, times, lats, lons, label='added', add_xy=True):

        if label in self.data:
            self.remove(label)

        if add_xy:
            eastings, northings = self.map_lonlat_to_xy(lons, lats)
            results = {'latitude': lats, 'longitude': lons, 'easting': eastings, 'northing': northings}
        else:
            results = {'latitude': lats, 'longitude': lons}

        xds = xr.Dataset()

        for key, value in results.items():
            xarr = xr.DataArray(data=value, coords=[times], dims=['time'])
            xds[key] = xarr

        self.data[label] = xds

    def add_dataset(self, dataset, label='added', add_xy=True):

        if label in self.data:
            self.remove(label)

        if add_xy:
            times = dataset['latitude']['time'].values
            lats = dataset['latitude'].values
            lons = dataset['longitude'].values
            self.add(times, lats, lons, label=label)

        else:
            self.data[label] = dataset

    def remove(self, label):
        del self.data[label]

    def map_lonlat_to_xy(self, lons, lats):
        xs, ys = self.map(lons, lats)
        return xs, ys

    def compute_distance_between_two_tracks(self, label1, label2, units='km'):

        data1 = self.data[label1]
        data2 = self.data[label2]

        if units == 'deg':
            x_key = 'longitude'
            y_key = 'latitude'

        elif units == 'm' or units == 'km':
            x_key = 'easting'
            y_key = 'northing'

        else:
            print('Invalid units. Options are: deg, m, and km.')
            x_key = None
            y_key = None

        x1s, y1s = data1[x_key], data1[y_key]
        x2s, y2s = data2[x_key], data2[y_key]

        stop_index = np.where(x1s['time'].values <= x2s['time'].values[-1])[0][-1]
        norms = np.empty(stop_index + 1)

        for i in range(stop_index + 1):

            t = x1s['time'][i]
            x1, y1 = x1s.values[i], y1s.values[i]
            x2, y2 = x2s.interp(time=t, assume_sorted=True).values, y2s.interp(time=t, assume_sorted=True).values
            norm = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            norms[i] = norm

        if units == 'km':
            norms /= 1000

        return norms

    def plot(self, keys, xy=False, **kwargs):

        tracks = []

        if xy:
            x_key, y_key = 'easting', 'northing'

        else:
            x_key, y_key = 'longitude', 'latitude'

        for key in keys:

            track = [self.data[key][y_key].values, self.data[key][x_key].values]
            tracks.append(track)

        fig, ax = plot.plot_track(*tracks, **kwargs)

        return fig, ax


class Simulator:

    def __init__(self, time_frame, start_location, start_velocity=(0, 0), **kwargs):
        """This class sets up and runs the components necessary to run an iceberg drift simulation.

        Args:
            time_frame (tuple of numpy.datetime64): start time, end time for the simulation.
            start_location (tuple of float): starting position (latitude, longitude) for the simulation.

        Kwargs:
            start_velocity (tuple of float): starting velocity (vx, vy) in m/s.
            time_step (numpy.timedelta64): Time step in seconds.
            drift_model (function): The drift model function.
            time_stepper (function): The numerical integrator function.
            ocean_model (str): Name of ocean model. Can be ECMWF or HYCOM.
            atmosphere_model (str): Name of the atmosphere model. Can be ECMWF or NARR.
            iceberg_size (str or tuple of float): size class for the iceberg or dims (waterline length, sail height).
            iceberg_shape (str): shape class for the iceberg.
        """

        self.start_location = start_location
        self.time_frame = time_frame
        self.start_velocity = start_velocity

        self.time_step = kwargs.pop('time_step', np.timedelta64(300, 's'))
        self.drift_model = kwargs.pop('drift_model', drift.newtonian_drift_wrapper)
        self.time_stepper = kwargs.pop('time_stepper', timesteppers.euler)

        self.ocean_model = kwargs.pop('ocean_model', 'ECMWF')
        self.atmosphere_model = kwargs.pop('atmosphere_model', 'NARR')

        self.ocean = metocean.Ocean(self.time_frame, model=self.ocean_model)
        self.atmosphere = metocean.Atmosphere(self.time_frame, model=self.atmosphere_model)

        self.iceberg_size = kwargs.pop('iceberg_size', 'LG')
        self.iceberg_shape = kwargs.pop('iceberg_shape', 'TAB')
        self.iceberg = iceberg.quickstart(self.time_frame[0], self.start_location, velocity=self.start_velocity,
                                          size=self.iceberg_size, shape=self.iceberg_shape)

        self.results = Results(**{'map': kwargs.pop('map', plot.get_map())})

    def set_constant_current(self, constants):
        self.ocean = metocean.Ocean(self.time_frame, model=self.ocean_model, constants=constants)

    def set_constant_wind(self, constants):
        self.atmosphere = metocean.Atmosphere(self.time_frame, model=self.ocean_model, constants=constants)

    def reload_ocean(self):
        self.ocean = metocean.Ocean(self.time_frame, model=self.ocean_model)

    def reload_atmosphere(self):
        self.atmosphere = metocean.Atmosphere(self.time_frame, model=self.atmosphere_model)

    def reload_iceberg(self):
        self.iceberg = iceberg.quickstart(self.time_frame[0], self.start_location, velocity=self.start_velocity,
                                          size=self.iceberg_size, shape=self.iceberg_shape)

    def run_simulation(self, label=None, **kwargs):
        """This method simulates iceberg drift.

        Args:
            label (str): Key by which the results of the simulation will be saved in results attribute.
        """

        if not self.iceberg.time == self.time_frame[0]:
            self.reload_iceberg()

        kwargs['time_step'] = self.time_step
        kwargs['time_stepper'] = self.time_stepper
        kwargs['drift_model'] = self.drift_model
        kwargs['ocean_model'] = self.ocean_model
        kwargs['atmosphere_model'] = self.atmosphere_model

        kwargs['ocean'] = self.ocean
        kwargs['atmosphere'] = self.atmosphere

        kwargs['iceberg'] = self.iceberg

        log = getLogger('{}'.format(strftime("%Y-%m-%d %H:%M:%S", gmtime())))
        file_handler = DebugFileHandler()
        log.addHandler(file_handler)
        log.setLevel(DEBUG)

        kwargs['log'] = log

        results = run_simulation(self.time_frame, self.start_location, self.start_velocity, **kwargs)

        del log

        if label is not None:
            self.results.add_dataset(results, label)
        else:
            return results

    def run_optimization(self, keys, x0, bounds, reference_vectors):
        """This function optimizes user specified drift simulation parameters using the Scipy minimize function.

        Args:
            keys (list of str): The names of the drift simulation kwargs to be optimized.
            x0 (numpy.ndarray): The initial guesses.
            bounds (list of list of float): The upper and lower bounds for the parameters being optimized.
            reference_vectors (tuple of xarray.core.dataarray.DataArray): The latitude, longitude vectors to compare to.

        Returns:
            optimization_result (scipy.optimize.optimize.OptimizeResult): Results from minimization.

        """
        f = self.optimization_wrapper
        optimization_result = minimize(f, x0=x0, bounds=bounds, args=(keys, reference_vectors))

        return optimization_result

    def optimization_wrapper(self, values, keys, reference_vectors):
        kwargs = dict(zip(keys, values))
        xds = run_simulation(self.time_frame, self.start_location, **kwargs)
        simulation_vectors = xds['latitude'], xds['longitude']
        square_errors = compute_norms(simulation_vectors, reference_vectors)
        mean_square_error = np.mean(square_errors)

        return mean_square_error


def compute_norms(simulation_vectors, reference_vectors, units='deg', **kwargs):

    sim_lats, sim_lons = deepcopy(simulation_vectors)
    ref_lats, ref_lons = deepcopy(reference_vectors)

    if units == 'm' or units == 'km':
        map_ = kwargs.pop('map', plot.get_map())
        sim_lons.values, sim_lats.values = map_(sim_lons.values, sim_lats.values)
        ref_lons.values, ref_lats.values = map_(ref_lons.values, ref_lats.values)

    stop_index = np.where(ref_lats['time'].values <= sim_lats['time'].values[-1])[0][-1]
    norms = np.empty(stop_index + 1)

    for i in range(stop_index + 1):

        time = ref_lats['time'][i]
        ref_lat = ref_lats[i].values
        ref_lon = ref_lons[i].values
        sim_lat = sim_lats.interp(time=time, assume_sorted=True).values
        sim_lon = sim_lons.interp(time=time, assume_sorted=True).values
        norm = np.sqrt((sim_lat - ref_lat) ** 2 + (sim_lon - ref_lon) ** 2)
        norms[i] = norm

    if units == 'km':
        norms /= 1000

    return norms


def run_simulation(time_frame, start_location, start_velocity=(0, 0), **kwargs):

    time_step = kwargs.pop('time_step', np.timedelta64(300, 's'))
    time_stepper = kwargs.pop('time_stepper', timesteppers.euler)
    drift_model = kwargs.pop('drift_model', drift.newtonian_drift_wrapper)
    ocean_model = kwargs.pop('ocean_model', 'ECMWF')
    atmosphere_model = kwargs.pop('atmosphere_model', 'NARR')

    start_time, end_time = time_frame
    dt = time_step.item().total_seconds()
    nt = int(np.timedelta64(end_time - start_time, 's').item().total_seconds() / dt)

    size = kwargs.pop('iceberg_size', 'LG')
    shape = kwargs.pop('iceberg_shape', 'TAB')
    iceberg_ = kwargs.pop('iceberg', iceberg.quickstart(start_time, start_location, velocity=start_velocity, size=size, shape=shape))

    current_constants = kwargs.pop('current_constants', None)
    wind_constants = kwargs.pop('wind_constants', None)

    ocean = kwargs.pop('ocean', metocean.Ocean(time_frame, model=ocean_model, constants=current_constants))
    atmosphere = kwargs.pop('atmosphere', metocean.Atmosphere(time_frame, model=atmosphere_model, constants=wind_constants))

    # Initialize arrays
    times = np.zeros(nt, dtype='datetime64[ns]')

    if drift_model is drift.newtonian_drift_wrapper:

        results = {'latitude': np.zeros(nt),
                   'longitude': np.zeros(nt),
                   'iceberg_eastward_velocity': np.zeros(nt),
                   'iceberg_northward_velocity': np.zeros(nt)}
        kwargs = {
            'form_drag_coefficient_in_air': kwargs.pop('Ca', iceberg_.FORM_DRAG_COEFFICIENT_IN_AIR),
            'form_drag_coefficient_in_water': kwargs.pop('Cw', iceberg_.FORM_DRAG_COEFFICIENT_IN_WATER),
            'skin_drag_coefficient_in_air': iceberg_.SKIN_DRAG_COEFFICIENT_IN_AIR,
            'skin_drag_coefficient_in_water': iceberg_.SKIN_DRAG_COEFFICIENT_IN_WATER,
            'sail_area': iceberg_.geometry.sail_area,
            'keel_area': iceberg_.geometry.keel_area,
            'top_area': iceberg_.geometry.waterline_length ** 2,
            'bottom_area': iceberg_.geometry.bottom_area,
            'mass': kwargs.pop('mass', iceberg_.geometry.mass),
            'latitude': iceberg_.latitude,
            'ekman': kwargs.pop('ekman', False),
            'depth_vec': kwargs.pop('depth_vec', np.arange(0, -110, -10)),
            'time_step': time_step,
            'eastward_current': ocean.current.eastward_velocities,
            'northward_current': ocean.current.northward_velocities,
            'eastward_wind': atmosphere.wind.eastward_velocities,
            'northward_wind': atmosphere.wind.northward_velocities,
            'log': kwargs.pop('log', None),
            'current_interpolator': ocean.current.interpolate,
            'wind_interpolator': atmosphere.wind.interpolate
        }

    else:
        results = {'latitude': np.zeros(nt),
                    'longitude': np.zeros(nt)}
        kwargs = {
            'form_drag_coefficient_in_air': kwargs.pop('Ca', iceberg_.FORM_DRAG_COEFFICIENT_IN_AIR),
            'form_drag_coefficient_in_water': kwargs.pop('Cw', iceberg_.FORM_DRAG_COEFFICIENT_IN_WATER),
            'waterline_length': iceberg_.geometry.waterline_length,
            'time_step': time_step,
            'eastward_current': ocean.current.eastward_velocities,
            'northward_current': ocean.current.northward_velocities,
            'eastward_wind': atmosphere.wind.eastward_velocities,
            'northward_wind': atmosphere.wind.northward_velocities,
            'current_interpolator': ocean.current.interpolate,
            'wind_interpolator': atmosphere.wind.interpolate
        }

    for i in range(nt):

        times[i] = iceberg_.time
        results['latitude'][i] = iceberg_.latitude
        results['longitude'][i] = iceberg_.longitude

        if drift_model is drift.newtonian_drift_wrapper:

            results['iceberg_eastward_velocity'][i] = iceberg_.eastward_velocity
            results['iceberg_northward_velocity'][i] = iceberg_.northward_velocity

        if time_stepper in (timesteppers.ab2, timesteppers.ab3):

            if drift_model is drift.newtonian_drift_wrapper:

                dx, dy, dvx, dvy = time_stepper(drift_model, dt,
                                                times[:i + 1],
                                                results['longitude'][:i + 1],
                                                results['latitude'][:i + 1],
                                                results['iceberg_eastward_velocity'][:i + 1],
                                                results['iceberg_northward_velocity'][:i + 1],
                                                **kwargs)
            else:

                dx, dy = time_stepper(drift_model, dt, times[:i+1], results['longitude'][:i+1], results['latitude'][:i+1], **kwargs)

        else:

            if drift_model is drift.newtonian_drift_wrapper:

                dx, dy, dvx, dvy = time_stepper(drift_model, dt,
                                                iceberg_.time, iceberg_.longitude, iceberg_.latitude,
                                                iceberg_.eastward_velocity, iceberg_.northward_velocity,
                                                **kwargs)

            else:

                dx, dy = time_stepper(drift_model, dt, iceberg_.time, iceberg_.longitude, iceberg_.latitude, **kwargs)

        if drift_model is drift.newtonian_drift_wrapper:

            iceberg_.eastward_velocity += dvx
            iceberg_.northward_velocity += dvy

        iceberg_.time += time_step
        iceberg_.latitude += tools.dy_to_dlat(dy)
        iceberg_.longitude += tools.dx_to_dlon(dx, iceberg_.latitude)

    xds = xr.Dataset()

    for key, value in results.items():
        xarr = xr.DataArray(data=value, coords=[times], dims=['time'])
        xds[key] = xarr

    return xds
