import type_methods
from utils import csvutils
import pkgutil
from utils.special_datetime import decadal_date as dateformat
from os.path import join
from math import isnan, sqrt, pow
import datetime
import numpy as np


class Dataset(object):
    """ Dataset Class which can handle different time formats (Decades, more to come...)

    Initialise with name of Datatype and Database
    """

    def __init__(self, datatype, database):
        self.method = MAPPING[datatype].methods
        self.database=database
        assert self.database is not None, 'Folder for specifed station does not exist!'
        self.datatype=datatype
        self.data = self.__read_data()
        self.years=sorted(list(self.data.viewkeys()))

    def __read_data(self):
        try:
            return(csvutils.load_data(join(self.database.filename, self.datatype + '.csv')))
        except:
            print "Dataset not found, create new"
            return self.__create_empty()

    def __create_empty(self):
            first_date=self.method.first_date()
            return {first_date.year: [float('nan')] * dateformat.datesperyear()}

    def get_feature(self, date, n=1, threshold=0, fillnan=False):
        """ Return timeseries as list from date backward in time for n time steps"""
        features=[]
        nan=0
        try:
            while len(features)<n:
                decadeofyear=date.decade_of_year
                year=date.year
                value=self.data[year][decadeofyear-1]
                if isnan(value):
                    nan=nan+1
                features.append(value)
                date=date.timedelta(-1)
        except:
            return [np.nan]
        if float(nan)/n>threshold:
            return [np.nan]
        else:
            features.reverse()
            features=np.array(features)
            if nan>0 and fillnan:
                nans, x=self.nan_helper(features)
                features[nans]=np.interp(x(nans),x(~nans),features[~nans])
            return features

    def nan_helper(self,x):
        return np.isnan(x), lambda z: z.nonzero()[0]

    def decadal_standard_deviation(self,decades_of_year):
        xi=[]
        for year in self.years:
            values=[]
            for decade in decades_of_year:
                values.append(self.data[year][decade-1])
            if any(np.isnan(values)):
                average=np.nan
            else:
                average=np.mean(values)
            if not isnan(average):
                xi.append(average)

        xm=self.decadal_average(decades_of_year)

        std_dev=0
        for x in xi:
            std_dev=std_dev+pow((x-xm),2)
        std_dev=sqrt(std_dev/(len(xi)-1))
        return std_dev

    def decadal_average(self,decades_of_year):
        values=[]
        for decade in decades_of_year:
            for year in self.years:
                values.append(self.data[year][decade-1])
        return np.nanmean(values)

    def update(self, date=None):
        ###### adapt date handling!!
        start_date=self.method.first_date()
        end_date=self.method.last_date()
        if date==None:
            end_date=dateformat.min(dateformat.today().timedelta(-1), end_date)
            start_date=dateformat.max(self.last_observation(), start_date)
        elif dateformat.decadal_difference(self.method.last_date(), date)>=0 and dateformat.decadal_difference(self.method.first_date(), date)<=0:
            end_date=date
            start_date=date
        else:
            return 'date is out of range for this datasource'

        if dateformat.decadal_difference(start_date, end_date)<=0:
            for date in self.decadal_daterange(start_date, end_date):
                #print date.year,date.decade_of_year
                value=self.method.get_observation(self.database,date)
                if date.year>self.years[-1]:
                    self.years.append(date.year)
                    self.data.update({date.year: [float('nan')] * dateformat.datesperyear()})
                self.data[date.year][date.decade_of_year-1]=value
            message='database has been updated'
        else:
            message='database is already up-to date'
        return message

    def write(self):
        csvutils.write_decadal_data(self.data, join(self.database.filename, self.datatype + '.xlsx'))

    def decadal_daterange(self,start_date, end_date):
        dates=[]
        for n in range(int(dateformat.decadal_difference(end_date, start_date)+1)):
            dates.append(start_date.timedelta(n))
        return dates

    def last_observation(self):
        lastyear=self.data[self.years[-1]]
        for i,e in reversed(list(enumerate(lastyear))):
            if isnan(lastyear[i]) and not isnan(lastyear[i-1]):
                return dateformat(self.years[-1], i + 1)
        if isnan(lastyear[-1]):
            return datetime.date(self.years[-1],1)
        else:
            return datetime.date(self.years[-1]+1,1)

    def max(self):
        max=-9999
        for date in dateformat.decadal_daterange(dateformat(self.years[0], 1), self.last_observation()):
            if self.data[date.year][date.decade_of_year-1]>max:
                max=self.data[date.year][date.decade_of_year-1]
        return max

    def transform2delta(self):
        datanew=dict()
        for year in self.years:
            datanew.update({year: [float('nan')] * dateformat.datesperyear()})

        for date in dateformat.decadal_daterange(dateformat(self.years[0], 2), self.last_observation()):
            datanew[date.year][date.decade_of_year-1]=self.data[date.year][date.decade_of_year-1]-self.data[date.timedelta(-1).year][date.timedelta(-1).decade_of_year-1]
        return TransformedDataset(self.datatype,self.years,datanew)

    def normalized(self):
        max=self.max()
        datanew = dict()
        for year in self.years:
            datanew.update({year: [float('nan')] * dateformat.datesperyear()})

        for date in dateformat.decadal_daterange(dateformat(self.years[0], 1), self.last_observation()):
            datanew[date.year][date.decade_of_year - 1] = self.data[date.year][date.decade_of_year - 1]/max
        return TransformedDataset(self.datatype, self.years, datanew)

class TransformedDataset(object):
    def __init__(self, datatype, years, data):
        self.datatype=datatype
        self.data = data
        self.years = years

    def get_feature(self, date, n=1, threshold=0):
        # gives back timeseries of n decades length
        features=[]
        nan=0
        try:
            while len(features)<n:
                decadeofyear=date.decade_of_year
                year=date.year
                value=self.data[year][decadeofyear-1]
                if isnan(value):
                    nan=nan+1
                features.append(value)
                date=date.timedelta(-1)
        except:
            return [np.nan]
        if float(nan)/n>threshold:
            return [np.nan]
        else:
            features.reverse()
            features = np.array(features)
            if nan > 0 and fillnan:
                nans, x = self.nan_helper(features)
                features[nans] = np.interp(x(nans), x(~nans), features[~nans])
            return features

    def nan_helper(self, x):
        return np.isnan(x), lambda z: z.nonzero()[0]

    def decadal_standard_deviation(self,decades_of_year):
        xi=[]
        for year in self.years:
            values=[]
            for decade in decades_of_year:
                values.append(self.data[year][decade-1])
            if any(np.isnan(values)):
                average=np.nan
            else:
                average=np.mean(values)
            if not isnan(average):
                xi.append(average)

        xm=self.decadal_average(decades_of_year)

        std_dev=0
        for x in xi:
            std_dev=std_dev+pow((x-xm),2)
        std_dev=sqrt(std_dev/(len(xi)-1))
        return std_dev

    def decadal_average(self,decades_of_year):
        values=[]
        for decade in decades_of_year:
            for year in self.years:
                values.append(self.data[year][decade-1])
        return np.nanmean(values)

    def decadal_daterange(self,start_date, end_date):
        dates=[]
        for n in range(int(dateformat.decadal_difference(end_date, start_date)+1)):
            dates.append(start_date.timedelta(n))
        return dates

    def last_observation(self):
        lastyear=self.data[self.years[-1]]
        for i,e in reversed(list(enumerate(lastyear))):
            if isnan(lastyear[i]) and not isnan(lastyear[i-1]):
                return dateformat(self.years[-1], i + 1)
        if isnan(lastyear[-1]):
            return datetime.date(self.years[-1],1)
        else:
            return datetime.date(self.years[-1]+1,1)

MAPPING = {
    'runoff': type_methods.decadalexceldata,
    'runoffdev': type_methods.decadalexceldata,
    'precip': type_methods.decadalexceldata,
    'temp': type_methods.decadalexceldata,
    'season': type_methods.decadalexceldata
}

if __name__ == '__main__':
    database=pkgutil.get_loader(join('database/chatkal'))
    data = Dataset('precip', database)
    withnans=data.get_feature(dateformat(2006,10),30,5)
    nonans=  data.get_feature(dateformat(2006,10),30,5,fillnan=True)
    import matplotlib.pyplot as plt
    plt.plot(withnans,'+',color='red')
    plt.plot(nonans,'.')
    plt.savefig('test.png')



    ##---------DATA AVAILIBILITY PLOT-----------
    # scadata = DailyDataset('sca',database)
    # runoffdata = DecadalDataset('runoff', database)
    # tempdata = DecadalDataset('temp', database)
    # precipdata = DecadalDataset('precip', database)
    # trmmdata = DailyDataset('precipitation', database)
    # trmmdata.update()
    # trmmdata.write()
    #
    # daterange=special_date.decadal_daterange(special_date(1930,1),special_date(2015,36))
    # import numpy
    # sca=numpy.empty(len(daterange))
    # runoff=numpy.empty(len(daterange))
    # temp=numpy.empty(len(daterange))
    # precip=numpy.empty(len(daterange))
    # trmm=numpy.empty(len(daterange))
    # datelist=numpy.full(len(daterange),None,dtype=datetime.date)
    #
    # for i,date in enumerate(daterange):
    #     sca[i] = 4 if (scadata.get_feature(date) is not None) else None
    #     runoff[i] = 1 if (runoffdata.get_feature(date) is not None) else None
    #     temp[i] = 2 if (tempdata.get_feature(date) is not None) else None
    #     precip[i] = 3 if (precipdata.get_feature(date) is not None) else None
    #     trmm[i] = 5 if (trmmdata.get_feature(date) is not None) else None
    #     datelist[i] = date.firstdate()
    #
    # from matplotlib import pyplot as plt
    # plt.figure(figsize=(15, 3), dpi=80)
    # plt.plot(datelist,sca,linewidth=20,solid_capstyle="butt")
    # plt.plot(datelist,runoff,linewidth=20,solid_capstyle="butt")
    # plt.plot(datelist,temp,linewidth=20,solid_capstyle="butt")
    # plt.plot(datelist,precip,linewidth=20,solid_capstyle="butt")
    # plt.plot(datelist,trmm,linewidth=20,solid_capstyle="butt")
    #
    # plt.ylim((0,6))
    # plt.xlim((datetime.date(1932,1,1),datetime.date(2015,12,31)))
    # plt.yticks([1,2,3,4,5],['runoff','temperatur\n(station)','precipitation\n(station)','MODIS Snow','TRMM precipitation'])
    # plt.draw()
    # plt.savefig('test.png')


