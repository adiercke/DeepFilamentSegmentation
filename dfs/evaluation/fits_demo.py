from astropy.coordinates import SkyCoord
from sunpy.map import Map, all_coordinates_from_map
from astropy import units as u
from matplotlib import pyplot as plt

import numpy as np

fits_file = '/net/reko/work1/soe/adiercke/ChroTel/chrotel_ha_lev1.0_20200728_114510.fits.gz'
resolution = 2048

s_map = Map(fits_file)
r_obs_pix = s_map.rsun_obs / s_map.scale[0]
scale_factor = resolution / (2 * r_obs_pix.value)
s_map = s_map.rotate(recenter=True, scale=scale_factor, missing=0, order=3)
arcs_frame = (resolution / 2) * s_map.scale[0].value
s_map = s_map.submap(SkyCoord(- arcs_frame * u.arcsec, - arcs_frame * u.arcsec, frame=s_map.coordinate_frame),
                     top_right=SkyCoord(arcs_frame * u.arcsec, arcs_frame * u.arcsec, frame=s_map.coordinate_frame))
pad_x = s_map.data.shape[0] - resolution
pad_y = s_map.data.shape[1] - resolution
s_map = s_map.submap([pad_x // 2, pad_y // 2] * u.pix,
                     [pad_x // 2 + resolution - 1, pad_y // 2 + resolution - 1] * u.pix)

# LDC
coords = all_coordinates_from_map(s_map)
radial_distance = ((np.sqrt(coords.Tx ** 2 + coords.Ty ** 2)) / s_map.rsun_obs).value
disk_filter = radial_distance <= 1
mu = np.cos(radial_distance * np.pi / 2)

plt.imshow(mu)
plt.colorbar()
plt.show()

plt.imshow(disk_filter)
plt.show()

map_data = s_map.data
map_dist = map_data[disk_filter]
mu_dist = mu[disk_filter]

mu_dist = mu_dist.flatten()
map_dist = map_dist.flatten()
fit = np.polyfit(mu_dist, map_dist, deg=4)
mu_fit = np.sum([fit[i] * mu ** i for i in reversed(range(5))], 0)
plt.scatter(mu_dist, map_dist)
plt.show()

s_map = Map(map_data / mu_fit * disk_filter, s_map.meta)
s_map.plot()
plt.show()
