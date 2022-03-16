import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from matplotlib.colors import Normalize
from sunpy.map import Map, all_coordinates_from_map


def get_data(fits_file):
    resolution = 1024
    s_map = Map(fits_file)
    r_obs_pix = s_map.rsun_obs / s_map.scale[0]
    scale_factor = resolution / (2 * r_obs_pix.value)
    s_map = s_map.rotate(recenter=True, scale=scale_factor, missing=0, order=3)
    arcs_frame = (resolution / 2) * s_map.scale[0].value
    s_map = s_map.submap(
        bottom_left=SkyCoord(- arcs_frame * u.arcsec, - arcs_frame * u.arcsec, frame=s_map.coordinate_frame),
        top_right=SkyCoord(arcs_frame * u.arcsec, arcs_frame * u.arcsec, frame=s_map.coordinate_frame))
    pad_x = s_map.data.shape[0] - resolution
    pad_y = s_map.data.shape[1] - resolution
    s_map = s_map.submap(bottom_left=[pad_x // 2, pad_y // 2] * u.pix,
                         top_right=[pad_x // 2 + resolution - 1, pad_y // 2 + resolution - 1] * u.pix)
    #
    # s_map.data /=  np.nanmedian(s_map.data)
    # LDC
    coords = all_coordinates_from_map(s_map)
    radial_distance = (np.sqrt(coords.Tx ** 2 + coords.Ty ** 2) / s_map.rsun_obs).value
    radial_distance[radial_distance >= 1] = np.NaN
    ideal_correction = np.cos(radial_distance * np.pi / 2)

    condition = np.logical_not(np.isnan(np.ravel(ideal_correction)))
    map_list = np.ravel(s_map.data)[condition]
    correction_list = np.ravel(ideal_correction)[condition]

    fit = np.polyfit(correction_list, map_list, 6)
    poly_fit = np.poly1d(fit)

    map_correction = poly_fit(ideal_correction)
    data = s_map.data / map_correction
    #
    data = Normalize(0.8, 1.3, clip=True)(data) * 2 - 1
    data = np.nan_to_num(data, nan=-1)
    #
    return np.array(data[None, :, :], dtype=np.float32)
