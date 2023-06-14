import os
import urllib.request

data_set_path = '/gpfs/gpfs0/robert.jarolim/data/filament/kso'
os.makedirs(data_set_path, exist_ok=True)

urls = ['http://cesar.kso.ac.at/halpha3a/2011/20111226/processed/kanz_halph_fi_20111226_090859.fts.gz',
        'http://cesar.kso.ac.at/halpha3a/2013/20131220/processed/kanz_halph_fi_20131220_111212.fts.gz',
        'http://cesar.kso.ac.at/halpha3a/2014/20141116/processed/kanz_halph_fi_20141116_072731.fts.gz',
        'http://cesar.kso.ac.at/halpha3a/2015/20151215/processed/kanz_halph_fi_20151215_074300.fts.gz',
        'http://cesar.kso.ac.at/halpha3a/2016/20161102/processed/kanz_halph_fi_20161102_072309.fts.gz',
        'http://cesar.kso.ac.at/halpha3a/2017/20171124/processed/kanz_halph_fi_20171124_070943.fts.gz',
        'http://cesar.kso.ac.at/halpha3a/2020/20201231/processed/kanz_halph_fi_20201231_122503.fts.gz']

for url in urls:
    urllib.request.urlretrieve(url, data_set_path + '/' + url.split('/')[-1])
