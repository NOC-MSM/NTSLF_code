"""

# Prototyping a surge ensemble plot

Ideas:
* All ensemble members as a paler thin line in the same colour, overlaid with a heavy darker line for the deterministic run
* Is transparency available? (If so you can use transparency to stack the ensemble members then more confidence will come out darker.)

* Vertical lines through HT events?
"""

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter
from matplotlib.cbook import get_sample_data

import glob, os
import numpy as np
import xarray as xr
from socket import gethostname
import datetime
from sftp_tools import Uploader


def get_latest_filename(tail:str='Z-surge_noc_det-surge.nc') -> str:
    """ Get latest file from list. E.g. *-surge_noc_det-surge.nc """
    list_of_files = glob.glob(dirname + "*"+tail)
    return max(list_of_files, key=os.path.getctime).split('/')[-1]

def timestamp_from_ds(ds:xr.DataArray) -> str:
    """ Get formatted timestamp from dataarray """
    try:
        return np.datetime_as_string(ds.forecast_reference_time.values, 'm')
    except:
        return "yyyy-mm-ddThh:00"

def timestamp_from_filename(filename:str) -> str:
    """
    filename like: "20220323T1200Z-surge_noc_det-ssh.nc"
    """
    if "/" in filename:
        print(f"Expecting filename without full path, not {filename}")
        return ""
    return datetime.datetime.strptime( filename.split('-')[0], '%Y%m%dT%H%MZ').strftime('%Y-%m-%dT%H:%MZ')


class Ensemble:
    def __init__(self,
                ds_ens:xr.DataArray=None, # ensemble dataarray
                ds_det:xr.DataArray=None, # deterministic dataarray
                station_id:int=9,
                 ):
        self.ds_ens = ds_ens
        self.ds_det = ds_det
        self.station_id = station_id
        self.process()

    def process(self):
        if self.ds_ens is not None:
            self.make_ensemble_line_plot()
            self.make_ensemble_line_plot_v2_totsealevel_ens()
            self.make_ensemble_line_plot_v3_alert()
        else:
            print(f"Need to load ensemble dataarray")


    def make_ensemble_line_plot_v2_totsealevel_ens(self):
        ds = self.ds_ens
        station_id = self.station_id

        station_str = (ds.station_name[station_id].data.astype('str')).flatten()[0]  # extract station name string from dataarray

        # Construct LineCollection to handle multiline plot of segments. Need to convert time into numbers for this function
        x = mdates.date2num(ds.time)
        ys = (self.ds_ens.zos_residual + self.ds_ens.zos_tide.broadcast_like(self.ds_ens.zos_residual)).isel(station=station_id).values
        z0 = -np.nanmin(ys) # fake z0
        ys = ys + z0

        segs = np.zeros((ds.dims['realization'], ds.dims['time'], 2))
        segs[:, :, 1] = ys
        segs[:, :, 0] = x

        line_segments = LineCollection(segs, linewidths=1, #linewidths=(0.5, 1, 1.5, 2),
                                       colors='grey', linestyle='solid', alpha=0.5)

        ## Plot it

        fig = plt.figure(figsize=(8, 6))
        # Add a gridspec with two rows and two columns and a ratio of 1 to 3 between
        # the size of the marginal axes and the main axes in both directions.
        # Also adjust the subplot parameters for a square plot.
        gs = fig.add_gridspec(2, 1,  height_ratios=(3, 1),
                              left=0.1, right=0.9, bottom=0.1, top=0.9,
                              hspace=0.1)
        # Create the Axes.
        ax0 = fig.add_subplot(gs[0])
        ax1 = fig.add_subplot(gs[1])#, sharex=ax0)

        ## Plot the ensemble ssh segments
        ax0.set_xlim(x.min(), x.max())
        ax0.set_ylim(np.nanmin(ys), np.nanmax(ys))
        ax0.add_collection(line_segments)

        # Add the tide
        ax0.plot( ds_ens.time,
                  ds_ens.zos_tide.isel(station=station_id).values
                  + z0, linewidth=0.5, color='k', linestyle='solid')

        # Add the deterministic forecast
        ax0.plot( ds_det.time,
                  (ds_det.zos_residual+ds_det.zos_tide).isel(station=station_id).values
                  + z0)

        ax0.set_ylabel('tide + surge (m)')

        day_format = DateFormatter("%a")
        ax0.xaxis.set_major_formatter(day_format)
        ax0.xaxis_date()

        ## Plot the surge ensemble
        # Construct LineCollection to handle multiline plot of segments. Need to convert time into numbers for this function
        x = mdates.date2num(ds.time)
        ys = self.ds_ens.zos_residual.isel(station=station_id).values

        segs = np.zeros((ds.dims['realization'], ds.dims['time'], 2))
        segs[:, :, 1] = ys
        segs[:, :, 0] = x

        line_segments = LineCollection(segs, linewidths=1,
                                       colors='grey', linestyle='solid', alpha=0.5)


        ## Plot the ensemble segments
        ax1.set_xlim(x.min(), x.max())
        ax1.set_ylim(np.nanmin(ys), np.nanmax(ys))
        ax1.add_collection(line_segments)
        # Add the deterministic forecast
        ax1.plot( ds_det.time,
                  ds_det.zos_residual.isel(station=station_id).values)
        ax1.set_ylabel('surge (m)')

        ax1.set_xlim(x.min(), x.max())

        date_format = DateFormatter("%d-%b-%Y")
        ax1.xaxis.set_major_formatter(date_format)
        ax1.xaxis_date()
        ax1.xaxis.set_tick_params(rotation=40)


        ## title
        suptitle_str = f"Ensemble sea level forecast for {station_str}"
        ens_timestamp_str = timestamp_from_ds(ds_ens)
        det_timestamp_str = timestamp_from_ds(ds_det)
        fig.suptitle(suptitle_str, fontsize=16, y=0.98) # Station title
        ax0.set_title(f"ensemble: {ens_timestamp_str}\ndeterministic: {det_timestamp_str}", fontsize=8)  # timestamp

        ## Met Office credit
        ax1.annotate(f"data source: Met Office",
                   xy=(ax1.get_xlim()[1] - 0.1, ax1.get_ylim()[0] ),
                   fontsize=6,
                   xycoords='data',
                   horizontalalignment='right',
                   verticalalignment='bottom')

        ## Logo
        im = plt.imread(get_sample_data(logo_file))
        axin = ax1.inset_axes([0.8, 0.12, 0.3, 0.3], zorder=2)
        axin.imshow(im)
        axin.axis('off')

        ## z0 note
        ax0.annotate(f"Issues:\n*fake z0\n*Used model tide\n*duplicate timestamps are for debugging\n*swap panels to align HT",
                   xy=(ax0.get_xlim()[0] + 0.1, ax0.get_ylim()[0] ),
                   fontsize=12,
                   xycoords='data',
                   horizontalalignment='left',
                   verticalalignment='bottom')

        #plt.show()

        ## OUTPUT FIGURES - svg
        fname = ofile.replace('.svg', '_' + str(station_id).zfill(4) + '_v2.svg')
        print(f"Save {fname}")
        fig.savefig(fname, transparent=True, bbox_inches='tight', pad_inches=0)
        plt.close(fig)

    def make_ensemble_line_plot_v3_alert(self):
        """ Alert level (m) - tide = limit of surge that induces flooding """
        alert_level = 4  # a function of location. Includes concept of z0. Alert level ~ Highest Astro Tide - z0 ~ 0.5*(Highest Astro Range)
        ds = self.ds_ens
        station_id = self.station_id

        station_str = (ds.station_name[station_id].data.astype('str')).flatten()[0]  # extract station name string from dataarray

        # Construct LineCollection to handle multiline plot of segments. Need to convert time into numbers for this function
        x = mdates.date2num(ds.time)
        ys = ds.zos_residual.isel(station=station_id).values

        segs = np.zeros((ds.dims['realization'], ds.dims['time'], 2))
        segs[:, :, 1] = ys
        segs[:, :, 0] = x

        # *colors* is sequence of rgba tuples.
        # *linestyle* is a string or dash tuple. Legal string values are
        # solid|dashed|dashdot|dotted.  The dash tuple is (offset, onoffseq) where
        # onoffseq is an even length tuple of on and off ink in points.  If linestyle
        # is omitted, 'solid' is used.
        # See `matplotlib.collections.LineCollection` for more information.
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

        line_segments = LineCollection(segs, linewidths=1, #linewidths=(0.5, 1, 1.5, 2),
                                       colors='grey', linestyle='solid', alpha=0.5)
                                       #colors=colors, linestyle='solid', alpha=0.05)

        ## Plot it

        fig = plt.figure(figsize=(8, 6))
        # Add a gridspec with two rows and two columns and a ratio of 1 to 3 between
        # the size of the marginal axes and the main axes in both directions.
        # Also adjust the subplot parameters for a square plot.
        gs = fig.add_gridspec(1, 1,
                              left=0.1, right=0.9, bottom=0.1, top=0.9,
                              hspace=0.1)
        # Create the Axes.
        ax0 = fig.add_subplot(gs[0])

        ## Plot the ensemble segments
        ax0.set_xlim(x.min(), x.max())
        ax0.set_ylim(np.nanmin(ys), np.nanmax(ys))
        ax0.add_collection(line_segments)
        # Add the deterministic forecast
        ax0.plot( ds_det.time, ds_det.zos_residual.isel(station=station_id).values)
        ax0.set_ylabel('surge (m)')

        day_format = DateFormatter("%a")
        ax0.xaxis.set_major_formatter(day_format)
        ax0.xaxis_date()

        ## Plot the Alert level (tide+hazard limit)
        alert = ax0.plot(x, alert_level - ds.zos_tide.isel(station=station_id).values, color='r')

        #date_format = DateFormatter("%d-%b-%Y")
        #ax1.xaxis.set_major_formatter(date_format)
        #ax1.xaxis_date()
        #ax1.xaxis.set_tick_params(rotation=40)


        ## title
        suptitle_str = f"Ensemble surge forecast for {station_str}"
        ens_timestamp_str = timestamp_from_ds(ds_ens)
        det_timestamp_str = timestamp_from_ds(ds_det)
        fig.suptitle(suptitle_str, fontsize=16, y=0.98) # Station title
        ax0.set_title(f"ensemble: {ens_timestamp_str}\ndeterministic: {det_timestamp_str}", fontsize=8)  # timestamp

        ## Met Office credit
        ax0.annotate(f"data source: Met Office",
                   xy=(ax0.get_xlim()[1] - 0.1, ax0.get_ylim()[0] ),
                   fontsize=6,
                   xycoords='data',
                   horizontalalignment='right',
                   verticalalignment='bottom')

        ## Logo
        im = plt.imread(get_sample_data(logo_file))
        axin = ax0.inset_axes([0.9, 0.03, 0.1, 0.1], zorder=2)
        axin.imshow(im)
        axin.axis('off')

        ## z0 note
        ax0.annotate(f"Alert level (fake threshold)", color='r',
                   xy=(ax0.get_xlim()[0] + 0.1,
                       ax0.get_ylim()[1] - 0.01*(ax0.get_ylim()[1]-ax0.get_ylim()[0]) ),
                   fontsize=12,
                   xycoords='data',
                   horizontalalignment='left',
                   verticalalignment='top')
        #plt.show()

        ## OUTPUT FIGURES - svg
        fname = ofile.replace('.svg', '_' + str(station_id).zfill(4) + '_v3.svg')
        print(f"Save {fname}")
        fig.savefig(fname, transparent=True, bbox_inches='tight', pad_inches=0)
        plt.close(fig)

    def make_ensemble_line_plot(self):
        ds = self.ds_ens
        station_id = self.station_id

        station_str = (ds.station_name[station_id].data.astype('str')).flatten()[0]  # extract station name string from dataarray

        # Construct LineCollection to handle multiline plot of segments. Need to convert time into numbers for this function
        x = mdates.date2num(ds.time)
        ys = ds.zos_residual.isel(station=station_id).values

        segs = np.zeros((ds.dims['realization'], ds.dims['time'], 2))
        segs[:, :, 1] = ys
        segs[:, :, 0] = x

        # *colors* is sequence of rgba tuples.
        # *linestyle* is a string or dash tuple. Legal string values are
        # solid|dashed|dashdot|dotted.  The dash tuple is (offset, onoffseq) where
        # onoffseq is an even length tuple of on and off ink in points.  If linestyle
        # is omitted, 'solid' is used.
        # See `matplotlib.collections.LineCollection` for more information.
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

        line_segments = LineCollection(segs, linewidths=1, #linewidths=(0.5, 1, 1.5, 2),
                                       colors='grey', linestyle='solid', alpha=0.5)
                                       #colors=colors, linestyle='solid', alpha=0.05)

        ## Plot it

        fig = plt.figure(figsize=(8, 6))
        # Add a gridspec with two rows and two columns and a ratio of 1 to 3 between
        # the size of the marginal axes and the main axes in both directions.
        # Also adjust the subplot parameters for a square plot.
        gs = fig.add_gridspec(2, 1,  height_ratios=(3, 1),
                              left=0.1, right=0.9, bottom=0.1, top=0.9,
                              hspace=0.1)
        # Create the Axes.
        ax0 = fig.add_subplot(gs[0])
        ax1 = fig.add_subplot(gs[1])#, sharex=ax0)

        ## Plot the ensemble segments
        ax0.set_xlim(x.min(), x.max())
        ax0.set_ylim(np.nanmin(ys), np.nanmax(ys))
        ax0.add_collection(line_segments)
        # Add the deterministic forecast
        ax0.plot( ds_det.time, ds_det.zos_residual.isel(station=station_id).values)
        ax0.set_ylabel('surge (m)')

        day_format = DateFormatter("%a")
        ax0.xaxis.set_major_formatter(day_format)
        ax0.xaxis_date()

        ## Plot the tide
        ax1.plot(x, ds.zos_tide.isel(station=station_id).values)
        ax1.set_xlim(x.min(), x.max())
        ax1.set_ylabel('tide (m)')

        date_format = DateFormatter("%d-%b-%Y")
        ax1.xaxis.set_major_formatter(date_format)
        ax1.xaxis_date()
        ax1.xaxis.set_tick_params(rotation=40)


        ## title
        suptitle_str = f"Ensemble surge forecast for {station_str}"
        ens_timestamp_str = timestamp_from_ds(ds_ens)
        det_timestamp_str = timestamp_from_ds(ds_det)
        fig.suptitle(suptitle_str, fontsize=16, y=0.98) # Station title
        ax0.set_title(f"ensemble: {ens_timestamp_str}\ndeterministic: {det_timestamp_str}", fontsize=8)  # timestamp

        ## Met Office credit
        ax0.annotate(f"data source: Met Office",
                   xy=(ax0.get_xlim()[1] - 0.1, ax0.get_ylim()[0] ),
                   fontsize=6,
                   xycoords='data',
                   horizontalalignment='right',
                   verticalalignment='bottom')

        ## Logo
        im = plt.imread(get_sample_data(logo_file))
        axin = ax0.inset_axes([0.9, 0.03, 0.1, 0.1], zorder=2)
        axin.imshow(im)
        axin.axis('off')


        #plt.show()

        ## OUTPUT FIGURES - svg
        fname = ofile.replace('.svg', '_' + str(station_id).zfill(4) + '.svg')
        print(f"Save {fname}")
        fig.savefig(fname, transparent=True, bbox_inches='tight', pad_inches=0)
        plt.close(fig)

#######
if __name__ == '__main__':

    if "LJOB" in gethostname().upper():  # Production job
        dirname = '/projectsa/surge_archive/surge_forecast/' # + str(datetime.datetime.now().year) + '/'
        fig_dir = '/projectsa/surge_archive/figures/ensembles_latest/'
        ofile = fig_dir + 'surge_ens_latest.svg'
        logo_file = fig_dir + '../NOC_Colour.png'
        filename_ens = get_latest_filename(tail='Z-surge_classa_ens-surge.nc')
        filename_det = "latest_surge_classa_det-surge.nc" #get_latest_filename(np.datetime64('now'), tail='Z-surge_classa_det-surge.nc')
    elif "LIVMAZ" in gethostname().upper():  # Debugging on local machine
        dirname = '/Users/jelt/Downloads/' + str(datetime.datetime.now().year) + '/'
        fig_dir = '/Users/jelt/Downloads/ensembles_latest/'
        ofile = fig_dir + 'surge_ens_latest.svg'
        filename_ens = "20230628T1800Z-surge_classa_ens-surge.nc"
        logo_file = '/Users/jelt/Library/CloudStorage/OneDrive-NOC/presentations/figures/logos/NOC_Colour.png'
        filename_det = "20230628T1800Z-surge_classa_det-surge.nc"
    else:
        print(f"Do not recognise hostname: {gethostname()}")


    station_id = 9 # Liverpool

    if(1): #try:
        ds_ens = xr.load_dataset(dirname + filename_ens)
        print(f'Processing {dirname + filename_ens}')

        ds_det = xr.load_dataset(dirname + filename_det)
        print(f'Processing {dirname + filename_det}')

    for station_id in range(47):
        ens = Ensemble(ds_ens=ds_ens, ds_det=ds_det, station_id=station_id)

    Uploader(local_dir=fig_dir,
             remote_dir="/local/users/ntslf/pub/ntslf_surge_animation/")
