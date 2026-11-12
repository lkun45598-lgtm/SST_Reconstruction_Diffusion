"""HomeyerRainbow colormap, reproduced from the official definition in cmweather 0.3.2 (cmweather/_cm_colorblind.py, BSD licence;
colormap developed by Cameron Homeyer with assistance from Bobby Jackson): 15 YUV nodes, linearly interpolated in RGB, 256 levels.
The function below is copied verbatim; nothing else of cmweather is needed."""
import numpy as np
from matplotlib import colors


def yuv_rainbow_24(nc):
    path1 = np.linspace(0.8 * np.pi, 1.8 * np.pi, nc)
    path2 = np.linspace(-0.33 * np.pi, 0.33 * np.pi, nc)

    y = np.concatenate(
        [np.linspace(0.3, 0.85, nc * 2 // 5), np.linspace(0.9, 0.0, nc - nc * 2 // 5)]
    )
    u = 0.40 * np.sin(path1)
    v = 0.55 * np.sin(path2) + 0.1

    rgb_from_yuv = np.array([[1, 0, 1.13983], [1, -0.39465, -0.58060], [1, 2.03211, 0]])
    cmap_dict = {'blue': [], 'green': [], 'red': []}
    for i in range(len(y)):
        yuv = np.array([y[i], u[i], v[i]])
        rgb = rgb_from_yuv.dot(yuv)
        red_tuple = (i / (len(y) - 1.0), rgb[0], rgb[0])
        green_tuple = (i / (len(y) - 1.0), rgb[1], rgb[1])
        blue_tuple = (i / (len(y) - 1.0), rgb[2], rgb[2])
        cmap_dict['blue'].append(blue_tuple)
        cmap_dict['red'].append(red_tuple)
        cmap_dict['green'].append(green_tuple)

    return cmap_dict


def homeyer_rainbow():
    return colors.LinearSegmentedColormap('HomeyerRainbow', yuv_rainbow_24(15), 256)


def homeyer_rainbow_soft(strength=0.25, lo=0.40, hi=0.72):
    """HomeyerRainbow-INSPIRED variant (NOT the official colormap): the official HomeyerRainbow with the saturation of its warm half
    (yellow -> orange -> red) reduced by up to `strength`, blended in smoothly between positions lo and hi. Hue and lightness untouched."""
    from matplotlib.colors import rgb_to_hsv, hsv_to_rgb, ListedColormap
    base = homeyer_rainbow()(np.linspace(0, 1, 256))[:, :3]; t = np.linspace(0, 1, 256)
    w = np.clip((t - lo) / (hi - lo), 0, 1); w = w * w * (3 - 2 * w)
    hsv = rgb_to_hsv(base); hsv[:, 1] *= 1 - strength * w
    return ListedColormap(hsv_to_rgb(hsv), name='HomeyerRainbow-inspired (soft warm)')


MUTED_ANCHORS = ['#285782', '#5686A5', '#91B8C8', '#C8D9D9', '#E9E4D6', '#E5B994', '#D5896C', '#B64F48']


def muted_sst():
    """Custom low-saturation oceanographic SST colormap (NOT an official/published colormap): 8 colour anchors, linear RGB interpolation, 256 levels.
    Blue (cold) -> pale blue/grey -> cream -> soft orange -> muted red (warm)."""
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list('MutedOceanographicSST', MUTED_ANCHORS, N=256)


VIVID_ANCHORS = ['#155A9C', '#238FC0', '#65C2C0', '#B5D995', '#F0E47A', '#F5B052', '#E97848', '#B83C35']


def vivid_sst():
    """Custom 'Vivid Oceanographic SST' colormap (NOT a published colormap): 8 anchors, linear RGB interpolation, 256 levels.
    Deep blue -> blue -> cyan -> light green -> soft yellow -> orange -> coral -> muted red."""
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list('VividOceanographicSST', VIVID_ANCHORS, N=256)
