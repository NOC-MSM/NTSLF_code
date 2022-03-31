"""
surge_anim.py

A python script to build an animation of the sea level anomaly
jelt 21Mar2022

A new environment was build as follows::

    module load anaconda/5-2021
    conda create -p /work/jelt/conda-env/ntslf_py39 python=3.9
    conda activate /work/jelt/conda-env/ntslf_py39
    conda install netCDF4 numpy xarray matplotlib
    conda install cartopy

If already built:
    module load anaconda/5-2021
    conda activate /work/jelt/conda-env/ntslf_py39

To run:
    python surge_anim.py

Know issues:
    There is a know warning regarding conversion of timestamp to daylight savings:
    "DeprecationWarning: parsing timezone aware datetimes is deprecated; this will raise an error in the future"
    There is currently no official / best solution.

"""


import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from numpy import pi
from matplotlib.cbook import get_sample_data
import matplotlib.ticker as ticker
import matplotlib as mpl
import os, glob
from math import cos, sin
import cartopy.crs as ccrs  # mapping plots
from cartopy.feature import NaturalEarthFeature
import xarray as xr
from socket import gethostname
import datetime
from datetime import timezone
import pytz

MIN_LAT = 48
MAX_LAT = 61
MIN_LON = -13
MAX_LON = 7

LIV_LAT = 53 + 24.5/60
LIV_LON = -(2+59.5/60)

SOT_LAT = 50 + 54/60 + 9/3600
SOT_LON = -(1+24/60+15/3600)

#################### INTERNAL FCTNS #########################

def dt64(now) -> np.datetime64:
    """ All the nano seconds can mess up conversions, so strip them out """
    return np.datetime64(np.datetime_as_string(now, unit="m"))

def get_am_pm(now) -> str:
    hour = now.astype(object).hour
    if (hour >= 12): return "pm"
    else: return "am"

def get_day(now) -> str:
    return now.astype(object).strftime('%a')

def timestamp_from_filename(filename:str) -> str:
    """
    filename like: "20220323T1200Z-surge_noc_det-ssh.nc"
    """
    if "/" in filename:
        print(f"Expecting filename without full path, not {filename}")
        return ""
    return datetime.datetime.strptime( filename.split('-')[0], '%Y%m%dT%H%MZ').strftime('%Y-%m-%dT%H:%MZ')

def to_localtime(now) -> np.datetime64:
    """ UTC --> np.datetime64(GMT/BST), str(GMT/BST) """
    datetime_obj_in = now.astype(object)
    datetime_obj_out = datetime_obj_in.astimezone(pytz.timezone("Europe/London"))
    if datetime_obj_out.dst() != datetime.timedelta(0,0): timezone_str = "BST"
    else: timezone_str = "GMT"
    return np.datetime64(datetime_obj_out), timezone_str

def get_filename_today(now, tail:str='T1200Z-surge_noc_det-surge.nc') -> str:
    """ E.g. 20220320T1200Z-surge_noc_det-surge.nc """
    return now.astype(object).strftime('%Y%m%d')+tail

def get_latest_surge_file() -> str:
    list_of_files = glob.glob(dirname+'*surge_noc_det-surge.nc') # * means all if need specific format then *.csv
    return max(list_of_files, key=os.path.getctime).split('/')[-1]

def get_latest_filename_today(now, tail:str='Z-surge_noc_det-surge.nc') -> str:
    """ Specify day but find hour. E.g. 20220320T*-surge_noc_det-surge.nc """
    list_of_file = glob.glob(dirname + now.astype(object).strftime('%Y%m%dT')+"????"+tail)
    return max(list_of_files, key=os.path.getctime).split('/')[-1]

def clock(ax, now):
    plt.ylim(0,1)
    plt.xlim(0,1)
    ax.text( 0.5, 0.01, get_am_pm(now), horizontalalignment='center', fontsize='9')
    ax.text( 0.5, 0.99, get_day(now), horizontalalignment='center', fontsize='9')
    hour= now.astype(object).hour
    minute = now.astype(object).minute
    second = now.astype(object).second
    angles_h= pi/2 - 2*pi*hour/12-2*pi*minute/(12*60)-2*second/(12*60*60)
    angles_m= pi/2 - 2*pi*minute/60-2*pi*second/(60*60)
    hand_m_r = 0.45
    hand_h_r = 0.3
    ax.plot([0.5, 0.5 + hand_h_r*cos(angles_h)], [0.5, 0.5 + hand_h_r*sin(angles_h)], color="black", linewidth=4)
    ax.plot([0.5, 0.5 + hand_m_r*cos(angles_m)], [0.5, 0.5 + hand_m_r*sin(angles_m)], color="black", linewidth=2)
    ax.grid(False)
    plt.axis('off')
    return ax

def create_geo_axes(lonbounds, latbounds, fig=None, ax=None):
    """
    A routine for creating an axis for any geographical plot. Within the
    specified longitude and latitude bounds, a map will be drawn up using
    cartopy. Any type of matplotlib plot can then be added to this figure.
    For example:

    Example Useage
    #############

        f,a = create_geo_axes(lonbounds, latbounds)
        sca = a.scatter(stats.longitude, stats.latitude, c=stats.corr,
                        vmin=.75, vmax=1,
                        edgecolors='k', linewidths=.5, zorder=100)
        f.colorbar(sca)
        a.set_title('SSH correlations \n Monthly PSMSL tide gauge vs CO9_AMM15p0',
                    fontsize=9)

    * Note: For scatter plots, it is useful to set zorder = 100 (or similar
            positive number)
    """

    # If no figure or ax is provided, create a new one
    if ax==None and fig==None:
        fig = plt.figure(1)
        fig.clf()
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

    coast = NaturalEarthFeature(category="physical", facecolor=[0.9, 0.9, 0.9], name="coastline", scale="50m")
    ax.add_feature(coast, edgecolor="gray")
    ax.set_aspect(1 / np.cos(np.deg2rad(np.mean(latbounds))))
    ax.set_xlim(lonbounds[0], lonbounds[1])
    ax.set_ylim(latbounds[0], latbounds[1])
    ax.grid(False)

    #plt.show()
    return fig, ax


def make_gif(files, output, delay=100, repeat=True, **kwargs):
    """
    Uses imageMagick to produce an animated .gif from a list of
    picture files.
    """

    loop = -1 if repeat else 0
    os.system('convert -delay %d -loop %d %s %s' % (delay, loop, " ".join(files), output))
    os.chmod(output, 0o755)  # -rwxr-xr-x


class Animate:
    def __init__(self,
                lon:xr.DataArray=None,
                lat:xr.DataArray=None,
                var:xr.DataArray=None,
                time:xr.DataArray=None,
                lon_bounds: [float, float] = None,
                lat_bounds: [float, float] = None,
                levels: list = [],
                title_str='Surge forecast (m)',
                cbar_str:str="",
                filename:str="",
                ofile:str=""
                 ):


        self.lon = lon
        self.lat = lat
        self.var = var
        self.time = time
        self.levels = levels
        self.title_str = title_str
        self.cbar_str = cbar_str


        #title_str = f'Surge forecast for {timestamp}'
        self.filename = filename
        self.ofile = ofile
        if lon_bounds == None: self.lon_bounds = [lon.min(), lon.max()]
        else: self.lon_bounds = lon_bounds
        if lat_bounds == None: self.lat_bounds = [lat.min(), lat.max()]
        else: self.lat_bounds = lat_bounds

        self.process()

    def process(self):
        files = []
        #for count in range(2):
        for count in range(len(self.time)):
            self.timestamp = np.datetime_as_string(dt64(self.time[count]), unit="m")
            f = self.make_frame(count=count)

            ## OUTPUT FIGURES
            fname = fig_dir + self.filename.replace('.nc', '_' + str(count).zfill(4) + '.png')
            print(count, fname)
            f.savefig(fname, dpi=100)

            files.append(fname)

        ## Make the animated gif and clean up the frame files
        make_gif(files, self.ofile, delay=20)
        for f in files:
            os.remove(f)

        ## Make a backup copy of gif if the max surge is large enough
        if "surge_anom_latest" in self.ofile and (self.var.max() > 1.0 or self.var.min() < -1.0):
            print(f'Backing up {self.ofile}')
            os.system(f'cp {self.ofile} {fig_dir + self.filename.replace(".nc", ".gif")}')



    def make_frame(self, count:int=0):

        cmap0 = plt.cm.get_cmap('PiYG_r', 256)
        cmap0.set_bad('#9b9b9b', 1.0)

        f, a = create_geo_axes(self.lon_bounds, self.lat_bounds)

        ## Contour fill surge + zero contour
        sca = a.contourf(self.lon, self.lat, self.var[count,:,:],
                         levels=self.levels,
                         cmap=cmap0)
        con = a.contour(self.lon, self.lat, self.var[count,:,:],
                        levels=[0],
                        linewidths=0.2,
                        zorder=100)

        ## title
        a.set_title(self.title_str, fontsize=8)

        ## Colorbar
        cax = f.add_axes([0.77, 0.12, 0.02, 0.76])  # RHS
        #cax = f.add_axes([0.23, 0.12, 0.02, 0.76])
        norm = mpl.colors.BoundaryNorm(self.levels, cmap0.N, extend='both')
        cbar=f.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap0),
                   cax=cax, orientation='vertical',
                   spacing='proportional',
                   label=self.cbar_str,
                   #format='% 1.1f',  # gives gap if +ve
                   )
        bare0 = lambda y, pos: ('%+g' if y > 0 else ('%-g' if y < 0 else '%g')) % y
        cbar.ax.yaxis.set_major_formatter(ticker.FuncFormatter(bare0))
        #cbar.ax.yaxis.set_ticks_position('left')
        cbar.ax.tick_params(labelsize=8)
        cbar.ax.tick_params(length=0)
        cbar.ax.yaxis.set_tick_params(pad=0)


        ## Met Office credit
        a.annotate('data source: Met Office',
                   xy=(self.lon_bounds[0] + 0.1, self.lat_bounds[0] + 0.1),
                   fontsize=6,
                   xycoords='data',
                   horizontalalignment='left',
                   verticalalignment='bottom'
                   )

        ## Logo
        im = plt.imread(get_sample_data(logo_file))
        # newax = f.add_axes([0.7, 0.12, 0.2, 0.2], zorder=1) ## lower right
        newax = f.add_axes([0.25, 0.08, 0.2, 0.2], zorder=1)  ## lower left
        newax.imshow(im)
        newax.axis('off')

        ## Clock
        # clock_ax = f.add_axes([0.52, 0.35, 0.1, 0.1], zorder=1)  ## over UK
        dt64_now, timezone_str = to_localtime(dt64(self.time[count]))
        clock_ax = f.add_axes([0.62, 0.18, 0.1, 0.1], zorder=1)  ## lower right
        clock(clock_ax, dt64_now)

        ## snapshot timestamp.
        snapshot_timestamp = np.datetime_as_string(dt64_now, unit="m").replace('T', ' ')+" "+timezone_str
        a.text(self.lon_bounds[1] - 0.1, self.lat_bounds[0] + 0.1, snapshot_timestamp,
               fontsize=6,
               horizontalalignment='right',
               verticalalignment='bottom'
               )

        ## simulation forecast timestamp
        #sim_timestamp = np.datetime_as_string(dt64(self.time[0]), unit="m").replace('T', 'Z')
        #a.text(self.lon_bounds[0] + 0.1, self.lat_bounds[1] - 0.1, "forecast start: "+sim_timestamp,
        #       fontsize=6,
        #       horizontalalignment='left',
        #       verticalalignment='top'
        #       )

        ## Liverpool
        a.plot([LIV_LON], [LIV_LAT], 'o', color='gray', markersize=4)
        a.annotate('Liverpool',
                   xy=(LIV_LON, LIV_LAT),
                   xytext=(LIV_LON, LIV_LAT),
                   textcoords='data',
                   fontsize=6,
                   horizontalalignment='left',
                   verticalalignment='top')

        ## Southampton
        a.plot([SOT_LON], [SOT_LAT], 'o', color='gray', markersize=4)
        a.annotate('Southampton',
                   xy=(SOT_LON, SOT_LAT),
                   xytext=(SOT_LON, SOT_LAT),
                   textcoords='data',
                   fontsize=6,
                   horizontalalignment='center',
                   verticalalignment='bottom')

        return f



if __name__ == '__main__':

    if "LJOB" in gethostname().upper():  # Production job
        dirname = '/projectsa/surge_archive/surge_forecast/'
        # filename_surge = '20220320T1200Z-surge_noc_det-surge.nc'
        # filename_ssh = "20220323T1200Z-surge_noc_det-ssh.nc"
        fig_dir = '/projectsa/surge_archive/figures/'
        ofile = fig_dir + 'surge_anom_latest.gif'
        logo_file = fig_dir + 'NOC_Colour.png'
        filename_surge = get_filename_today(np.datetime64('now'), tail='T1200Z-surge_noc_det-surge.nc')
        filename_ssh = get_filename_today(np.datetime64('now'), tail='T1200Z-surge_noc_det-ssh.nc')
        filename_ssh = get_latest_filename_today(np.datetime64('now'), tail='Z-surge_noc_det-ssh.nc')
    elif "LIVMAZ" in gethostname().upper():  # Debugging on local machine
        dirname = '/Users/jeff/Downloads/'
        fig_dir = dirname
        ofile = fig_dir + 'surge_anom_latest.gif'
        filename_surge = '20220327T0600Z-surge_noc_det-surge.nc'
        logo_file = '/Users/jeff/Documents/presentations/figures/logos/NOC_Colour.png'
        filename_ssh = "20220323T1200Z-surge_noc_det-ssh.nc"

    else:
        print(f"Do not recognise hostname: {gethostname()}")



    try:
        #filename = get_filename_today(np.datetime64('now'), tail='T1200Z-surge_noc_det-surge.nc')  # update filename
        #filename_surge = get_latest_surge_file()  # update filename
        ds = xr.load_dataset(dirname + filename_surge)
        print(f'Processing {dirname + filename_surge}')
        plt.rcParams["text.usetex"] = True  # To enable latex interpreter used in title string

        animate = Animate(lon=ds.longitude,
                          lat = ds.latitude,
                          var=ds.zos_residual,
                          time=ds.time,
                          levels=[-1, -0.7, -0.3, -0.1, 0, 0.1, 0.3, 0.7, 1],
                          title_str=r"{{\Large Surge forecast (m)}}" + '\n' + f"{timestamp_from_filename(filename_surge)}",
                          cbar_str = "",
                          filename=filename_surge,
                          ofile=fig_dir+'surge_anom_latest.gif')
    except:
        print(f'Filename: {filename_surge} not processed')



    try:
        #filename = get_filename_today(np.datetime64('now'), tail='T1200Z-surge_noc_det-ssh.nc')  # update filename
        #filename = get_latest_surge_file()  # update filename
        ds = xr.load_dataset(dirname + filename_ssh)
        print(f'Processing {dirname + filename_ssh}')

        animate = Animate(lon=ds.longitude,
                          lat = ds.latitude,
                          var=ds.zos,
                          time=ds.time,
                          levels=[-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5],
                          title_str=r"{{\Large Sea level forecast (m)}}" + '\n' + f"{timestamp_from_filename(filename_ssh)}",
                          #cbar_str="total water level (m)",
                          cbar_str="",
                          filename=filename_ssh,
                          ofile=fig_dir+'ssh_latest.gif')

    except:
        print(f'Filename: {filename_ssh} not processed')
