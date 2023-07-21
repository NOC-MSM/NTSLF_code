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
    conda install pysftp
    conda install fileinput

If already built:
    module load anaconda/5-2021
    conda activate /work/jelt/conda-env/ntslf_py39

To run:
    python surge_anim.py

Know issues:
    None

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
from sftp_tools import Uploader
import fileinput
plt.rcParams['svg.fonttype'] = 'none'

MIN_LAT = 46.55
MAX_LAT = 61
MIN_LON = -13
MAX_LON = 9.5

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
    datetime_obj_in = now.astype(object).replace(tzinfo=timezone.utc)
    datetime_obj_out = datetime_obj_in.astimezone(pytz.timezone("Europe/London"))
    if datetime_obj_out.dst() != datetime.timedelta(0,0): timezone_str = "BST"
    else: timezone_str = "GMT"
    return np.datetime64(datetime_obj_out.replace(tzinfo=None)), timezone_str

def get_filename_today(now, tail:str='T1200Z-surge_noc_det-surge.nc') -> str:
    """ E.g. 20220320T1200Z-surge_noc_det-surge.nc """
    return now.astype(object).strftime('%Y%m%d')+tail

def get_latest_surge_file() -> str:
    list_of_files = glob.glob(dirname+'*surge_noc_det-surge.nc') # * means all if need specific format then *.csv
    return max(list_of_files, key=os.path.getctime).split('/')[-1]

def get_latest_filename_today(now, tail:str='Z-surge_noc_det-surge.nc') -> str:
    """ Specify day but find hour. E.g. 20220320T*-surge_noc_det-surge.nc """
    list_of_files = glob.glob(dirname + now.astype(object).strftime('%Y%m%dT')+"????"+tail)
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

def create_geo_axes(lonbounds, latbounds,
                    fig=None, ax=None,
                    projection=ccrs.PlateCarree(),
                    data_crs=ccrs.PlateCarree()):
    """
    A routine for creating an axis for any geographical plot. Within the
    specified longitude and latitude bounds, a map will be drawn up using
    cartopy. Any type of matplotlib plot can then be added to this figure.

    Accommodates coordinate transforms from data's coordinates (data_crs) onto
    a new coordinate system (projection).

    For example:

    Example Useage
    ##############

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
        fig = plt.figure()#figsize=(5,7))
        fig.clf()
        ax = fig.add_subplot(1, 1, 1, projection=projection)

    ax.set_extent([lonbounds[0], lonbounds[1], latbounds[0], latbounds[1]],
                  crs=data_crs)

    coast = NaturalEarthFeature(category="physical", facecolor=[0.9, 0.9, 0.9], name="coastline", scale="50m")
    ax.add_feature(coast, edgecolor="gray")
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
                suptitle_str = '',
                cbar_str:str="",
                cmap_str:str="PiYG_r",
                filename:str="",
                ofile_gif:str="",
                ofile_svg:str="",
                 ):


        self.lon = lon
        self.lat = lat
        self.var = var
        self.time = time
        self.levels = levels
        self.title_str = title_str
        self.suptitle_str = suptitle_str
        self.cbar_str = cbar_str
        self.cmap_str = cmap_str
        self.proj = ccrs.Mercator()  # coord sys for projected (displayed) data
        self.data_crs = ccrs.PlateCarree() # coord sys of data

        self.filename = filename
        self.ofile_gif = ofile_gif
        self.ofile_svg = ofile_svg
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
            f = self.make_frame(count=count, cmap_str=self.cmap_str) #cmap_str="PiYG_r")

            ## OUTPUT FIGURES - svg
            fname = self.ofile_svg.replace('.svg', '_' + str(count).zfill(4) + '.svg')
            print(count, fname)
            self.save_svg(fig=f, fname=fname)
            plt.close(f)

            files.append(fname)

        ## Make the animated gif and clean up the frame files
        make_gif(files, self.ofile_gif, delay=20)
        for f in files:
            pass #os.remove(f)

        ## Make a backup copy of gif if the max surge is large enough
        if "surge_anom_latest" in self.ofile_gif and (self.var.max() > 1.0 or self.var.min() < -1.0):
            print(f'Backing up {self.ofile_gif}')
            os.system(f'cp {self.ofile_gif} {fig_dir + self.filename.replace(".nc", ".gif")}')


    def save_svg(self, fig, fname:str):
        fig.savefig(fname, transparent=True, bbox_inches='tight', pad_inches=0)
        os.system(f'scour --set-precision=5  -i {fname} -o {fname.replace(".svg","_v2.svg")}')
        os.system(f'mv {fname.replace(".svg","_v2.svg")} {fname}')

        #with fileinput.FileInput(fname, inplace=True, backup='.bak') as file:
        with fileinput.FileInput(fname, inplace=True) as file:
            for line in file:
                new_line = line \
                    .replace("width=\"277.24pt\" height=\"300.67pt\"", "") \
                    .replace("font=\"6px 'sans-serif'\"", "font-size=\"6px\" font-family=\"sans-serif\"") \
                    .replace("font=\"8px 'sans-serif'\"", "font-size=\"8px\" font-family=\"sans-serif\"") \
                    .replace("font=\"9px 'sans-serif'\"", "font-size=\"9px\" font-family=\"sans-serif\"") \
                    .replace("font=\"16px 'sans-serif'\"", "font-size=\"16px\" font-family=\"sans-serif\"")
                if "Logo placeholder" in new_line:
                    new_line = "<image href=\"https://noc.ac.uk/files/logos/logo2023-noc.svg\" width=\"40\" height=\"40\" x=\"190\" y=\"252\"/>"
                print(new_line, end='') # remove image size spec

    def make_frame(self, count:int=0, cmap_str="PiYG_r"):

        cmap0 = plt.cm.get_cmap(cmap_str, 256)
        #cmap0.set_bad('#9b9b9b', 1.0)

        f, a = create_geo_axes(self.lon_bounds, self.lat_bounds,
                               projection=self.proj,
                               data_crs=self.data_crs)

        ## Contour fill surge + zero contour
        sca = a.contourf(self.lon, self.lat, self.var[count,:,:],
                         levels=self.levels,
                         cmap=cmap0,
                         transform=self.data_crs)
        con = a.contour(self.lon, self.lat, self.var[count,:,:],
                        levels=[0],
                        linewidths=0.2,
                        zorder=100,
                        transform=self.data_crs)

        ## title
        f.suptitle(self.suptitle_str, fontsize=16, y=0.98) # label
        a.set_title(f'Forecast produced at: {self.title_str}', fontsize=8)  # timestamp

        ## Colorbar
        cax = f.add_axes([0.78, 0.12, 0.02, 0.76])  # RHS
        norm = mpl.colors.BoundaryNorm(self.levels, cmap0.N, extend='both')
        cbar=f.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap0),
                   cax=cax, orientation='vertical',
                   spacing='proportional',
                   #format='% 1.1f',  # gives gap if +ve
                   )
        cbar.set_label(self.cbar_str, rotation=90, fontsize=6)
        bare0 = lambda y, pos: ('%+g' if y > 0 else ('%-g' if y < 0 else '% g')) % y
        cbar.ax.yaxis.set_major_formatter(ticker.FuncFormatter(bare0))
        #cbar.ax.yaxis.set_ticks_position('left')
        cbar.ax.tick_params(labelsize=8)
        cbar.ax.tick_params(length=0)
        cbar.ax.yaxis.set_tick_params(pad=2)




        ## Logo
        if(0):
            im = plt.imread(get_sample_data(logo_file))
            # newax = f.add_axes([0.7, 0.12, 0.2, 0.2], zorder=1) ## lower right
            newax = f.add_axes([0.27, 0.14, 0.1, 0.1], zorder=1)  ## lower left
            newax.imshow(im)
            newax.axis('off')
        else:  ## Logo placeholder
            a.annotate('Logo placeholder',
                   xy=(self.lon_bounds[0] + 0.1, self.lat_bounds[0] + 0.1),
                   fontsize=6,
                   xycoords = self.data_crs._as_mpl_transform(a),
                   horizontalalignment='left',
                   verticalalignment='bottom')
        ## Met Office credit
        a.annotate('data source: Met Office',
                   xy=(self.lon_bounds[1] - 0.1, self.lat_bounds[0] + 0.1),
                   fontsize=6,
                   xycoords = self.data_crs._as_mpl_transform(a),
                   #xycoords='data',
                   horizontalalignment='right',
                   verticalalignment='bottom')

        ## Clock
        # clock_ax = f.add_axes([0.52, 0.35, 0.1, 0.1], zorder=1)  ## over UK
        dt64_now, timezone_str = to_localtime(dt64(self.time[count]))
        #clock_ax = f.add_axes([0.65, 0.15, 0.1, 0.1], zorder=1)  ## lower right
        clock_ax = f.add_axes([0.65, 0.30, 0.1, 0.1], zorder=1)  ## mid lower right
        clock(clock_ax, dt64_now)

        ## snapshot timestamp.
        snapshot_timestamp = np.datetime_as_string(dt64_now, unit="m").replace('T', ' ')+" "+timezone_str
        a.annotate(snapshot_timestamp,
                   xy=(self.lon_bounds[1] - 0.1, 50.0),
                   xycoords = self.data_crs._as_mpl_transform(a),
                   fontsize=6,
                   horizontalalignment='right',
                   verticalalignment='bottom'
                   )

        ## Liverpool
        a.plot([LIV_LON], [LIV_LAT], 'o', color='gray', markersize=4, alpha=0.6,
               transform=self.data_crs)
        a.annotate('Liverpool',
                   xy=(LIV_LON, LIV_LAT),
                   xytext=(LIV_LON, LIV_LAT),
                   xycoords=self.data_crs._as_mpl_transform(a),#'data',
                   textcoords=self.data_crs._as_mpl_transform(a),#'data',
                   fontsize=6,
                   horizontalalignment='left',
                   verticalalignment='top')

        ## Southampton
        a.plot([SOT_LON], [SOT_LAT], 'o', color='gray', markersize=4, alpha=0.6,
               transform=self.data_crs)
        a.annotate('Southampton',
                   xy=(SOT_LON, SOT_LAT),
                   xytext=(SOT_LON, SOT_LAT),
                   xycoords=self.data_crs._as_mpl_transform(a),#'data',
                   textcoords=self.data_crs._as_mpl_transform(a),#'data',
                   family='sans-serif',
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
        #ofile_gif = fig_dir + 'surge_anom_latest.gif'
        #ofile_svg = fig_dir + 'surge_anom_latest.svg'
        logo_file = fig_dir + 'NOC_Colour.png'
        #filename_surge = get_filename_today(np.datetime64('now'), tail='T1200Z-surge_noc_det-surge.nc')
        filename_surge = get_latest_filename_today(np.datetime64('now'), tail='Z-surge_noc_det-surge.nc')
        #filename_ssh = get_filename_today(np.datetime64('now'), tail='T1200Z-surge_noc_det-ssh.nc')
        filename_ssh = get_latest_filename_today(np.datetime64('now'), tail='Z-surge_noc_det-ssh.nc')
    elif "LIVMAZ" in gethostname().upper():  # Debugging on local machine
        dirname = '/Users/jelt/Downloads/'
        fig_dir = dirname
        #ofile_gif = fig_dir + 'surge_anom_latest.gif'
        #ofile_svg = fig_dir + 'surge_anom_latest.svg'
        filename_surge = '20220327T0600Z-surge_noc_det-surge.nc'
        logo_file = '/Users/jelt/Library/CloudStorage/OneDrive-NOC/presentations/figures/logos/NOC_Colour.png'
        filename_ssh = "20220323T1200Z-surge_noc_det-ssh.nc"

    else:
        print(f"Do not recognise hostname: {gethostname()}")

    try:
        ds = xr.load_dataset(dirname + filename_surge)
        print(f'Processing {dirname + filename_surge}')

        animate = Animate(lon=ds.longitude,
                          lat = ds.latitude,
                          var=ds.zos_residual,
                          time=ds.time,
                          lon_bounds=[MIN_LON, MAX_LON],
                          lat_bounds=[MIN_LAT, MAX_LAT],
                          levels=[-1, -0.7, -0.3, -0.1, 0, 0.1, 0.3, 0.7, 1],
                          suptitle_str = "Surge forecast (m)",
                          title_str = timestamp_from_filename(filename_surge),
                          cbar_str = "relative to modelled tide",
                          cmap_str = "PiYG_r",
                          filename=filename_surge,
                          ofile_svg=fig_dir+'surge_anom_latest/surge_anom_latest.svg',
                          ofile_gif=fig_dir+'surge_anom_latest.gif')

    except:
        print(f'Filename: {filename_surge} not processed')

    try:
        Uploader(local_dir=fig_dir+"surge_anom_latest/",
                 remote_dir="/local/users/ntslf/pub/ntslf_surge_animation/")
    except:
        print(f'sftp upload from {fig_dir+"ssh_latest/"} failed')


    try:
        ds = xr.load_dataset(dirname + filename_ssh)
        print(f'Processing {dirname + filename_ssh}')

        animate = Animate(lon=ds.longitude,
                          lat = ds.latitude,
                          var=ds.zos,
                          time=ds.time,
                          lon_bounds=[MIN_LON, MAX_LON],
                          lat_bounds=[MIN_LAT, MAX_LAT],
                          levels=[-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5],
                          suptitle_str= "Sea level forecast (m)",
                          title_str= timestamp_from_filename(filename_ssh),
                          cbar_str="relative to model datum",
                          cmap_str="bwr", # "seismic",
                          filename=filename_ssh,
                          ofile_svg=fig_dir+'ssh_latest/ssh_latest.svg',
                          ofile_gif=fig_dir+'ssh_latest.gif')

    except:
        print(f'Filename: {filename_ssh} not processed')

    try:
        Uploader(local_dir=fig_dir+"ssh_latest/",
                 remote_dir="/local/users/ntslf/pub/ntslf_surge_animation/")
    except:
        print(f'sftp upload from {fig_dir+"ssh_latest/"} failed')

