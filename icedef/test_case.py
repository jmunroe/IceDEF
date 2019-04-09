#import numpy as np
#import xarray as xr
import pandas as pd
#import matplotlib.pyplot as plt
from icedef import metocean
from icedef import statoil_arcticnet_data as sd
from icedef.plot import *
from astroML.stats import fit_bivariate_normal, bivariate_normal
from matplotlib.patches import Ellipse


class TestCase:

    def __init__(self, beacon_id='20498', start_time=None, end_time=None):

        self.filename = '0' + beacon_id + '0' + '_2015.csv'
        self.path = sd.beacon_dir_path
        self.df = sd.get_beacon_df(self.path + self.filename, start_time=start_time, end_time=end_time)
        self.ref_berg = sd.create_ref_berg_from_df(self.df)
        self.ref_times = self.ref_berg.history['time']
        self.ref_lats = xr.DataArray(self.ref_berg.history['latitude'], coords=[self.ref_times], dims=['time'])
        self.ref_lons = xr.DataArray(self.ref_berg.history['longitude'], coords=[self.ref_times], dims=['time'])
        self.start_time = np.datetime64(self.df['DataDate_UTC'].values[0])
        self.end_time = np.datetime64(self.df['DataDate_UTC'].values[len(self.df)-1])
        self.time_frame = self.start_time, self.end_time
        self.start_latitude = self.df.Latitude[0]
        self.start_longitude = self.df.Longitude[0]
        self.start_location = self.start_latitude, self.start_longitude
        self.start_velocity = sd.get_iceberg_velocity_from_dataframe(self.df, 0, 1)

        self.observation_bounding_box = {'time': (self.time_frame[0] - np.timedelta64(30, 'D'),
                                                  self.time_frame[1] + np.timedelta64(30, 'D')),
                                         }
        # self.avos_ds = tools.subset_ds(sd.get_avos_ds(), self.observation_bounding_box)
        # self.adcp_ds = tools.subset_ds(sd.get_adcp_ds(), self.observation_bounding_box)
        # self.wind_velocity_df = self.get_wind_velocity_df()
        # self.current_velocity_df = self.get_current_velocity_df()
        self.avos_ds = xr.open_dataset('../notebooks/avos_ds.nc')
        self.adcp_ds = xr.open_dataset('../notebooks/adcp_ds.nc')
        self.wind_velocity_df = pd.read_csv('../notebooks/wind_velocity_df.csv').drop(columns='Unnamed: 0')
        self.current_velocity_df = pd.read_csv('../notebooks/current_velocity_df.csv').drop(columns='Unnamed: 0')

        self.wind_distribution = self.get_bivariate_distribution_of_velocity_corrections(self.wind_velocity_df)
        self.current_distribution = self.get_bivariate_distribution_of_velocity_corrections(self.current_velocity_df)

    def plot_track(self, **kwargs):

        fig, ax = plot_track([self.ref_lats.values, self.ref_lons.values], **kwargs)

    def get_wind_velocity_df(self):

        narr_atm = metocean.Atmosphere((self.avos_ds.time.values[0], self.avos_ds.time.values[-1]))
        wind_velocity_df = pd.DataFrame(columns=['t', 'x', 'y', 'u', 'v', 'iu', 'iv'])
        len_ = len(self.avos_ds.time.values)
        for i in range(len_):
            t = self.avos_ds.time[i].values
            x = self.avos_ds.longitude[i].values
            y = self.avos_ds.latitude[i].values
            u = self.avos_ds.eastward_velocity[i].values
            v = self.avos_ds.northward_velocity[i].values
            if np.isfinite(u):
                if np.isfinite(v):
                    if -60 <= x <= -40:
                        if 40 <= y <= 60:
                            iu, iv = narr_atm.wind.interpolate((t, y, x))
                            wind_velocity_df.loc[len(wind_velocity_df)] = [t, x, y, float(u), float(v), float(iu), float(iv)]

        return wind_velocity_df

    def get_current_velocity_df(self):

        ecmwf_ocean = metocean.Ocean((self.adcp_ds.time.values[0], self.adcp_ds.time.values[-1]), model='ECMWF')
        current_velocity_df = pd.DataFrame(columns=['t', 'x', 'y', 'u', 'v', 'iu', 'iv'])
        len_ = len(self.adcp_ds.time.values)
        for i in range(len_):
            t = self.adcp_ds.time.values[i]
            x = self.adcp_ds.longitude.values[i]
            y = self.adcp_ds.latitude.values[i]
            u = self.adcp_ds.eastward_velocity.values[i, 0]
            v = self.adcp_ds.northward_velocity.values[i, 0]
            if np.isfinite(u):
                if np.isfinite(v):
                    if -60 <= x <= -40:
                        if 40 <= y <= 60:
                            iu, iv = ecmwf_ocean.current.interpolate((t, y, x))
                            current_velocity_df.loc[len(current_velocity_df)] = [t, x, y, float(u), float(v), float(iu), float(iv)]

        return current_velocity_df

    def get_bivariate_distribution_of_velocity_corrections(self, df):

        U1 = df.u.values
        V1 = df.v.values
        U2 = df.iu.values
        V2 = df.iv.values

        fit = fit_bivariate_normal(U1 - U2, V1 - V2)

        return fit

    def sample_wind_distribution(self, size=1):

        return bivariate_normal(*self.wind_distribution, size=size).ravel()

    def sample_current_distribution(self, size=1):

        return bivariate_normal(*self.current_distribution, size=size).ravel()

    def plot_distributions(self, filename=None, current_only=False, wind_only=False):

        fig = plt.figure()

        for i, df in enumerate([self.current_velocity_df, self.wind_velocity_df]):

            U1 = df.u.values
            V1 = df.v.values
            U2 = df.iu.values
            V2 = df.iv.values

            if wind_only:
                i = 1

            if i == 0:

                params = self.current_distribution

            else:

                params = self.wind_distribution

            ellipse1 = Ellipse(xy=params[0], width=2 * params[1], height=2 * params[2], angle=np.rad2deg(params[3]),
                               alpha=0.8, edgecolor='b', lw=4, facecolor='none')
            ellipse2 = Ellipse(xy=params[0], width=2 * 2 * params[1], height=2 * 2 * params[2],
                               angle=np.rad2deg(params[3]),
                               alpha=0.8, edgecolor='b', lw=4, facecolor='none')

            if i == 0:

                if current_only:

                    subplot_number = 111

                else:

                    subplot_number = 121

                ax = fig.add_subplot(subplot_number, aspect='equal')
                ax.set_title('Current Correction Distribution')

            else:

                if wind_only:

                    subplot_number = 111

                else:

                    subplot_number = 122

                ax = fig.add_subplot(subplot_number, aspect='equal')
                ax.set_title('Wind Correction Distribution')

            ax.axhline(y=0, color='grey')
            ax.axvline(x=0, color='grey')
            ax.scatter(*params[0], color='k', zorder=2)
            ax.annotate(s=f'{np.round(params[0], 2)}', xy=(params[0][0], params[0][1]))

            ax.scatter(U1 - U2, V1 - V2, color='r')

            ax.add_artist(ellipse1)
            ax.add_artist(ellipse2)
            ax.set_xlabel('dU (m/s)')
            ax.set_ylabel('dV (m/s)')
            i += 1

            if current_only:
                break

        fig.tight_layout()

        if filename:
            fig.savefig(filename, bbox_inches='tight')


class TestCaseA(TestCase):

    START_TIME = np.datetime64('2015-05-06T15:27:39')
    END_TIME = np.datetime64('2015-05-07T06:25:51')
    BEACON_ID = '90679'

    def __init__(self, add_timedelta=np.timedelta64(0, 'D')):
        self.START_TIME += add_timedelta
        self.END_TIME = self.START_TIME + np.timedelta64(1, 'D')
        super().__init__(beacon_id=self.BEACON_ID, start_time=self.START_TIME, end_time=self.END_TIME)


class TestCaseB(TestCase):

    START_TIME = np.datetime64('2015-04-24T22:53:29')
    END_TIME = START_TIME + np.timedelta64(3, 'D')
    BEACON_ID = '20498'

    def __init__(self, add_timedelta=np.timedelta64(0, 'D')):
        self.START_TIME += add_timedelta
        self.END_TIME = self.START_TIME + np.timedelta64(3, 'D')
        super().__init__(beacon_id=self.BEACON_ID, start_time=self.START_TIME, end_time=self.END_TIME)


class TestCaseB6(TestCase):

    START_TIME = np.datetime64('2015-04-24T22:53:29')
    END_TIME = START_TIME + np.timedelta64(6, 'D')
    BEACON_ID = '20498'

    def __init__(self, add_timedelta=np.timedelta64(0, 'D')):
        self.START_TIME += add_timedelta
        self.END_TIME = self.START_TIME + np.timedelta64(6, 'h')
        super().__init__(beacon_id=self.BEACON_ID, start_time=self.START_TIME, end_time=self.END_TIME)


class TestCaseB12(TestCase):

    START_TIME = np.datetime64('2015-04-24T22:53:29')
    END_TIME = START_TIME + np.timedelta64(12, 'D')
    BEACON_ID = '20498'

    def __init__(self, add_timedelta=np.timedelta64(0, 'D')):
        self.START_TIME += add_timedelta
        self.END_TIME = self.START_TIME + np.timedelta64(12, 'h')
        super().__init__(beacon_id=self.BEACON_ID, start_time=self.START_TIME, end_time=self.END_TIME)


class TestCaseB24(TestCase):

    START_TIME = np.datetime64('2015-04-24T22:53:29')
    END_TIME = START_TIME + np.timedelta64(24, 'D')
    BEACON_ID = '20498'

    def __init__(self, add_timedelta=np.timedelta64(0, 'D')):
        self.START_TIME += add_timedelta
        self.END_TIME = self.START_TIME + np.timedelta64(24, 'h')
        super().__init__(beacon_id=self.BEACON_ID, start_time=self.START_TIME, end_time=self.END_TIME)


class TestCaseC(TestCase):

    START_TIME = np.datetime64('2015-04-23T18:48:37') #np.datetime64('2015-04-29') #np.datetime64('2015-04-24T15:16:06') #
    END_TIME = START_TIME + np.timedelta64(3, 'D')
    BEACON_ID = '50519'

    def __init__(self, add_timedelta=np.timedelta64(0, 'D')):
        self.START_TIME += add_timedelta
        self.END_TIME = self.START_TIME + np.timedelta64(3, 'D')
        super().__init__(beacon_id=self.BEACON_ID, start_time=self.START_TIME, end_time=self.END_TIME)


class TestCaseC6(TestCase):

    START_TIME = np.datetime64('2015-04-23T18:48:37') #np.datetime64('2015-04-29') #np.datetime64('2015-04-24T15:16:06') #
    END_TIME = START_TIME + np.timedelta64(6, 'h')
    BEACON_ID = '50519'

    def __init__(self, add_timedelta=np.timedelta64(0, 'D')):
        self.START_TIME += add_timedelta
        self.END_TIME = self.START_TIME + np.timedelta64(6, 'h')
        super().__init__(beacon_id=self.BEACON_ID, start_time=self.START_TIME, end_time=self.END_TIME)


class TestCaseC12(TestCase):

    START_TIME = np.datetime64('2015-04-23T18:48:37') #np.datetime64('2015-04-29') #np.datetime64('2015-04-24T15:16:06') #
    END_TIME = START_TIME + np.timedelta64(12, 'h')
    BEACON_ID = '50519'

    def __init__(self, add_timedelta=np.timedelta64(0, 'D')):
        self.START_TIME += add_timedelta
        self.END_TIME = self.START_TIME + np.timedelta64(12, 'h')
        super().__init__(beacon_id=self.BEACON_ID, start_time=self.START_TIME, end_time=self.END_TIME)


