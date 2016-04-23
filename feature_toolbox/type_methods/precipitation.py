import datetime
from struct import unpack
from urllib2 import Request, urlopen
from os.path import join

import numpy
#from osgeo import ogr
from retrying import retry


class methods(object):
    start_observation = datetime.date(1998, 1, 1)
    end_observation = datetime.date(2015, 12, 30)
    server_url_TRMM = 'ftp://disc3.nascom.nasa.gov/data/s4pa/TRMM_L3/TRMM_3B42_daily/YYYY/DOY/3B42_daily.YYYY.MM.DD.7.bin'
    tmp = '/home/jules/TRMMtmp'

    @classmethod
    def first_date(cls):
        return cls.start_observation

    @classmethod
    def last_date(cls):
        return cls.end_observation

    @classmethod
    def get_observation(cls, database, date):

        if date < cls.first_date() or date > cls.last_date():
            return float('nan')

        url = cls.server_url_TRMM
        doy = ((date - datetime.timedelta(1)) - datetime.date((date - datetime.timedelta(1)).year, 1, 1)).days + 1
        url = url.replace('/YYYY', '/' + str((date - datetime.timedelta(1)).year))
        url = url.replace('.YYYY', '.' + str(date.year))
        url = url.replace('.MM', '.' + str(date.month).zfill(2))
        url = url.replace('.DD', '.' + str(date.day).zfill(2))
        url = url.replace('/DOY', '/' + str(doy).zfill(3))

        trmmfile_path = cls.download(url)
        if trmmfile_path == None:
            return float('nan')

        driver = ogr.GetDriverByName('ESRI Shapefile')
        inDataSet = driver.Open(join(database.filename, 'catchment.shp'))
        inLayer = inDataSet.GetLayer()
        bbox = inLayer.GetExtent()  # lonmin,lonmax,latmin,latmax

        lonmin = bbox[0]
        lonmax = bbox[1]
        latmin = bbox[2]
        latmax = bbox[3]

        TRMMobject = TRMM_data(trmmfile_path)
        value = TRMMobject.evaluate(lonmin, lonmax, latmin, latmax)
        del TRMMobject
        return value

    @classmethod
    @retry(stop_max_attempt_number=3, wait_exponential_multiplier=1000, wait_exponential_max=10000)
    def download(cls, url):
        try:
            filename = join(cls.tmp, url[-27:])
            try:
                local_file = open(filename, 'rb')
                if not local_file.read(1):
                    raise Exception
                #print(filename)
            except:
                local_file = open(filename, 'wb')
                req = Request(url)
                response = urlopen(req)
                local_file.write(response.read())
                #print(url)
            local_file.close
            return filename
        except:
            raise
            return None

    @classmethod
    def save_tempfile(self, binaryfile):
        temp = tempfile.NamedTemporaryFile(delete=False)
        temp.write(binaryfile.read())
        temp.close()
        return temp.name


class TRMM_data(object):
    def __init__(self, datapath):
        self.data = self.read_trmm_bin(datapath)

    def read_trmm_bin(self, BinaryFile):

        f = open(BinaryFile, 'rb')

        NumbytesFile = 576000
        NumElementxRecord = -1440
        data = numpy.zeros((400, 1440))

        for PositionByte in range(NumbytesFile, 0, NumElementxRecord):
            Record = []
            for c in range(PositionByte - 720, PositionByte, 1):
                f.seek(c * 4)
                DataElement = unpack('>f', f.read(4))
                Record.extend(DataElement)
            for c in range(PositionByte - 1440, PositionByte - 720, 1):
                f.seek(c * 4)
                DataElement = unpack('>f', f.read(4))
                Record.extend(DataElement)
            data[PositionByte / 1440 - 1, :] = numpy.asarray(Record)
        f.close()
        from matplotlib import pyplot as plot
#        plot.imshow(data)
#        plot.colorbar()
#        plot.clim(0,180)
#        plot.show()
#        plot.draw()
        return data

    def evaluate(self, lonmin, lonmax, latmin, latmax):
        imin = round(self.lon2floatindex(lonmin),0)
        imax = round(self.lon2floatindex(lonmax),0)
        jmin = round(self.lat2floatindex(latmin),0)
        jmax = round(self.lat2floatindex(latmax),0)

        a = self.data[jmin:jmax+1, imin:imax+1].flat
        return numpy.mean([n for n in a if n>=0])

    def lon2floatindex(self, lon):
        return float(lon * 4 + 719.5)

    def lat2floatindex(self, lat):
        return float(lat * 4 + 199.5)
