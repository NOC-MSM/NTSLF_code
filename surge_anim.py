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

"""


import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from numpy import pi
from matplotlib.cbook import get_sample_data
import matplotlib as mpl
import os, glob
from math import cos, sin
import cartopy.crs as ccrs  # mapping plots
from cartopy.feature import NaturalEarthFeature
import xarray as xr
from socket import gethostname

if "LJOB" in gethostname().upper():  # Production job
    dirname  = '/projectsa/surge_archive/surge_forecast/'
    filename = '20220320T1200Z-surge_noc_det-surge.nc'
    fig_dir   = '/projectsa/surge_archive/figures/'
    ofile     = fig_dir + 'surge_anom_latest.gif'
    logo_file = fig_dir + 'NOC_Colour.png'
elif "LIVMAZ" in gethostname().upper():  # Debugging on local machine
    dirname = '/Users/jeff/Downloads/'
    fig_dir = dirname
    ofile     = fig_dir + 'surge_anom_latest.gif'
    filename = '20220320T1200Z-surge_noc_det-surge.nc'
    logo_file = '/Users/jeff/Documents/presentations/figures/logos/NOC_Colour.png'
else:
    print(f"Do not recognise hostname: {gethostname()}")


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

def get_filename_today(now) -> str:
    """ E.g. 20220320T1200Z-surge_noc_det-surge.nc """
    tail = 'T1200Z-surge_noc_det-surge.nc'
    return now.astype(object).strftime('%Y%m%d')+tail

def get_latest_surge_file() -> str:
    list_of_files = glob.glob(dirname+'*surge_noc_det-surge.nc') # * means all if need specific format then *.csv
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




# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    #filename = get_filename_today( np.datetime64('now') )  # update filename
    try:
        filename = get_latest_surge_file()  # update filename
    except:
        print('Use default filename: {filename}')
    print(dirname + filename)
    ds = xr.load_dataset(dirname+filename)

    MIN_LON = ds.longitude.min()
    MAX_LON = ds.longitude.max()
    MIN_LAT = ds.latitude.min()
    MAX_LAT = ds.latitude.max()

    min_val = ds.zos_residual.min()
    max_val = ds.zos_residual.max()

    levels = [-1, -0.7, -0.3, -0.1, 0, 0.1, 0.3, 0.7, 1]

    #levels = [-5,-4,-3,-2,-1,0,1,2,3,4,5]
    extreme = max(abs(np.min(levels)), abs(np.max(levels)))
    #max_lat = ds.latitude.max().values
    files   = []


    ## Create colorbar and nan color
    #cmap0 = plt.cm.get_cmap('Spectral_r', 256)
    cmap0 = plt.cm.get_cmap('PiYG_r', 256)
    cmap0.set_bad('#9b9b9b', 1.0)

    for count in range(len(ds.time)):
        #for count in range(5):
        ## Create figure and axes for plot
        f, a = create_geo_axes([MIN_LON, MAX_LON], [MIN_LAT, MAX_LAT])

        ## Contour fill surge + zero contour
        sca = a.contourf(ds.longitude, ds.latitude, ds.zos_residual[count,:,:],
                        levels=levels,
                        #vmin=-extreme, vmax=extreme,
                        cmap=cmap0)
        #sca.set_clim([-extreme, extreme])
        con = a.contour(ds.longitude, ds.latitude, ds.zos_residual[count,:,:],
                  levels=[0],
                  linewidths=0.2,
                  zorder=100)

        ## title and timestamp
        timestamp = np.datetime_as_string(dt64(ds.time[count]), unit="m")
        a.set_title(f'Surge forecast for {timestamp}',
                    fontsize=12)

        ## Colorbar
        cax =f.add_axes([0.77, 0.12, 0.02, 0.76])
        norm = mpl.colors.BoundaryNorm(levels, cmap0.N, extend='both')
        #cbar = mpl.colorbar.ColorbarBase( cax, mpl.cm.ScalarMappable(norm=norm, cmap=cmap0) )
        f.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap0),
                     cax=cax, orientation='vertical',
                     spacing='proportional',
                     label="total water level - tide, in metres")
        #cbar = mpl.colorbar.ColorbarBase(cax, orientation='horizontal',
        #                            cmap=cmap0,
        #                            norm=mpl.colors.Normalize(-extreme, extreme),
        #                            extend='both'
        #                            )
        #cbar.set_label('total water level - tide, in metres')

        ## simulation timestamp
        sim_str = filename.split('-')[0]
        sim_timestamp = np.datetime64( sim_str[0:4] + '-' + sim_str[4:6] + '-' + sim_str[6:8] + "T" + sim_str[9:11] + ':' + sim_str[11:13])
        a.text(MIN_LON+0.1, MAX_LAT-0.1, sim_timestamp,
               fontsize=6,
               horizontalalignment='left',
               verticalalignment='top'
               )

        ## Met Office credit
        #a.text(MIN_LON+0.5, MIN_LAT+0.1, 'data source: Met Office',
        a.annotate( 'data source: Met Office',
               xy=(0.03, 0.0),
               fontsize=6,
               xycoords = 'axes fraction',
               horizontalalignment='left',
               verticalalignment='bottom'
               )

        ## Logo
        im = plt.imread(get_sample_data(logo_file))
        #newax = f.add_axes([0.7, 0.12, 0.2, 0.2], zorder=1) ## lower right
        newax = f.add_axes([0.25, 0.08, 0.2, 0.2], zorder=1) ## lower left
        newax.imshow(im)
        newax.axis('off')

        ## Clock
        #clock_ax = f.add_axes([0.52, 0.35, 0.1, 0.1], zorder=1)  ## over UK
        clock_ax = f.add_axes([0.62, 0.18, 0.1, 0.1], zorder=1)  ## lower right
        clock(clock_ax, dt64(ds.time[count]))

        ## Liverpool
        a.plot([LIV_LON], [LIV_LAT], 'o', color='gray', markersize=4)
        a.annotate('Liverpool',
                    xy=(LIV_LON, LIV_LAT),
                    xytext=(LIV_LON, LIV_LAT),
                    textcoords='data',
                    fontsize=6,
                    horizontalalignment='left',
                    verticalalignment='bottom')


        ## Southampton
        a.plot([SOT_LON], [SOT_LAT], 'o', color='gray', markersize=4)
        a.annotate('Southampton',
                    xy=(SOT_LON, SOT_LAT),
                    xytext=(SOT_LON, SOT_LAT),
                    textcoords='data',
                    fontsize=6,
                    horizontalalignment='left',
                    verticalalignment='bottom')


        ## OUTPUT FIGURES
        fname = fig_dir + filename.replace('.nc', '_' + str(count).zfill(4) + '.png')
        print(timestamp, fname)
        f.savefig(fname, dpi=100)
        plt.close()

        files.append(fname)

    # Make the animated gif and clean up the files
    make_gif(files, ofile, delay=20)

    for f in files:
        os.remove(f)

    ## Make a backup copy of gif if the max surge is large enough
    if ds.zos_residual.max() > 1.0 or ds.zos_residual.min() < -0.1:
        os.system(f'cp {ofile} {fig_dir+filename.replace(".nc",".gif")}')
