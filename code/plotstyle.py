"""SST map style = project CLAUDE.md "SST 场图绘制规范" (user's reference figure). Colours are eyeballed from the
reference screenshot (cmap RdYlBu_r, land tan, missing gray)."""
import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.ticker import FuncFormatter, MultipleLocator

LAND, MISSING = '#D6C6A2', '#B8B8B8'
EXTENT = (111.0, 118.0, 15.0, 24.0)           # lon0, lon1, lat0, lat1 (north-up)
LAT_T, LON_T = [15, 17, 19, 21, 23], [111, 113, 115, 117]


def sst_cmap():
    c = plt.get_cmap('RdYlBu_r').copy(); c.set_bad((0, 0, 0, 0)); return c   # transparent: gray_base + tan land stay visible


def err_cmap():
    c = plt.get_cmap('Reds').copy(); c.set_bad((0, 0, 0, 0)); return c


def compose(field_c, ocean, valid=None):
    """field in degC; NaN where ocean pixel has no value to show (gray). Land -> sentinel handled by draw()."""
    f = np.where(ocean, field_c, np.nan).astype('float32')
    if valid is not None: f = np.where(valid, f, np.nan)
    return f


def draw(ax, field_c, land, cmap, vmin, vmax):
    """Call gray_base(ax, land) first. Layers: gray (missing ocean) < tan (land) < field (NaN transparent)."""
    ax.imshow(np.where(land, 1.0, np.nan), extent=EXTENT, origin='upper', interpolation='nearest',
              cmap=ListedColormap([LAND]), vmin=0, vmax=1)
    im = ax.imshow(np.where(land, np.nan, field_c), extent=EXTENT, origin='upper',
                   interpolation='nearest', cmap=cmap, vmin=vmin, vmax=vmax)
    return im


def gray_base(ax, land):
    ax.imshow(np.where(land, np.nan, 1.0), extent=EXTENT, origin='upper', interpolation='nearest',
              cmap=ListedColormap([MISSING]), vmin=0, vmax=1)


def style_axes(ax, bottom, left):
    ax.set_xlim(EXTENT[0], EXTENT[1]); ax.set_ylim(EXTENT[2], EXTENT[3])
    ax.set_xticks(LON_T); ax.set_yticks(LAT_T)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:.0f}°E'))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:.0f}°N'))
    ax.tick_params(direction='out', labelbottom=bottom, labelleft=left, labelsize=11)
    for sp in ax.spines.values(): sp.set_linewidth(0.8); sp.set_color('black')
    ax.set_aspect('equal')
    if bottom: ax.set_xlabel('Longitude', fontsize=12)
    if left: ax.set_ylabel('Latitude', fontsize=12)


def title(ax, line1, line2=None, size=13):
    ax.set_title(line1 + ('\n' + line2 if line2 else ''), fontsize=size, fontweight='bold', family='DejaVu Sans')


def shared_colorbar(fig, axes, im, label, step=0.5, shrink=0.5):
    cb = fig.colorbar(im, ax=list(np.ravel(axes)), location='right', shrink=shrink, pad=0.03, aspect=22)
    cb.set_label(label, fontsize=14)
    cb.ax.yaxis.set_major_locator(MultipleLocator(step))
    cb.ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:.1f}'))
    cb.ax.tick_params(labelsize=12)
    return cb


def cbar_slot(fig, slot, im, label, step=0.5, box=(0.3, 0.25, 0.4, 0.5)):
    """Colorbar in its own gridspec slot (vertically centred inset), for hand-built layouts. box = x0,y0,w,h in slot fraction."""
    holder = fig.add_subplot(slot); holder.axis('off')
    cax = holder.inset_axes(list(box))
    cb = fig.colorbar(im, cax=cax); cb.set_label(label, fontsize=13)
    cb.ax.yaxis.set_major_locator(MultipleLocator(step))
    cb.ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:.1f}'))
    cb.ax.tick_params(labelsize=11)
    return cb
