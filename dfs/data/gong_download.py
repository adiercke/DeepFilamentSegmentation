import logging
import os
import warnings
from datetime import timedelta, datetime
from multiprocessing import Pool
from urllib.request import urlopen

import pandas as pd
from astropy.io.fits import HDUList
from sunpy.map import Map

data_set_path = '/gpfs/gpfs0/robert.jarolim/data/filament/gong'
os.makedirs(data_set_path, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    handlers=[
        logging.StreamHandler()
    ])

# load gong data set
gong_df = pd.read_csv('/gpfs/gpfs0/robert.jarolim/data/gong_overview_2010_2021.csv', parse_dates=['datetime'],
                      index_col=0)
gong_df = gong_df[~gong_df['location'].isin(['Ah', 'Zh'])]
times = gong_df.datetime.dt.round('5min')
gong_df['diff'] = gong_df.datetime.sub(times).abs()
# grouped_df = gong_df.groupby(times)
# merged_series = gong_df.loc[grouped_df['diff'].idxmin()]
# merged_series.index = merged_series.datetime.dt.round('5min')


# download time frames
###########################################################
def downloadData(df):
    warnings.simplefilter('ignore')
    map_path = os.path.join(data_set_path, '%s.fits') % df.name.to_pydatetime().isoformat('T')
    if os.path.exists(map_path):
        print('exists')
        return map_path
    try:
        datetime = df.datetime.to_pydatetime()
        url = datetime.strftime('https://gong2.nso.edu/HA/haf/%Y%m/%Y%m%d/%Y%m%d%H%M%S') + '%s.fits.fz' % df.location
        url_request = urlopen(url, timeout=5)
        fits_data = url_request.read()
        hdul = HDUList.fromstring(fits_data)
        hdul.verify('silentfix')
        #
        header = hdul[1].header
        header['cunit1'] = 'arcsec'
        header['cunit2'] = 'arcsec'
        header['ctype1'] = 'SOLAR_X'
        header['ctype2'] = 'SOLAR_Y'
        #
        s_map = Map(hdul[1].data, header)
        s_map.save(map_path)
        print('downloaded', map_path)
        return map_path
    except Exception as ex:
        logging.error(str(ex))

download_files = []
locations = ['Bh', 'Ch', 'Th', 'Uh', 'Lh', 'Mh']
hours = [18, 14, 10, 6, 2, 22]
for l, h in zip(locations, hours):
    location_df = gong_df[gong_df['location'] == l]
    grouped_df = location_df.groupby(times)
    location_df = location_df.loc[grouped_df['diff'].idxmin()]
    location_df.index = location_df.datetime.dt.round('5min')
    td = timedelta(days=1)
    start_time = datetime(2010, 1, 1, h)
    end_time = datetime.now()
    download_times = [start_time + i * td for i in range((end_time - start_time) // td)]
    location_files = location_df[location_df.index.isin(download_times)]
    download_files += [location_files]
    print('Found %d out of %d dates' % (len(location_files), len(download_times)))

download_files = pd.concat(download_files)

# sync download
# [downloadData(df) for i, df in download_files.iterrows()]
# async download
with Pool(4) as pool:
    _ = pool.map(downloadData, [df for i, df in download_files.iterrows()])
