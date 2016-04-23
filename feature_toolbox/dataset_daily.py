import type_methods
from utils import csvutils
import pkgutil
import datetime
from os.path import join
from math import isnan
from numpy import mean,nan
from utils.special_datetime import decadal_date as dateformat



class DailyDataset(object):
    def __init__(self, datatype, database):
        self.method = MAPPING[datatype].methods
        self.database=database
        assert self.database is not None, 'Folder for specifed station %s does not exist!' %(self.station)
        self.datatype=datatype
        self.data = self.__read_data()
        self.years=sorted(list(self.data.viewkeys()))

    def __read_data(self):
        try:
            return(csvutils.load_data(join(self.database.filename,self.datatype+'.csv')))
        except:
            return self.__create_empty()

    def __create_empty(self):
            first_date=self.method.first_date()
            return {first_date.year: [float('nan')] * 366}

    def get_dailyfeature(self, date, n=1):
        # gives back timeseries of n days length
        year=date.year
        doy=(date-datetime.date(year,1,1)).days+1
        #doy=(date.lastdate()-datetime.date(year,1,1)).days+1
        try:
            if doy <= n:
                daysoy = 366 if self.leapyear(year-1) else 365
                feature=self.data[year-1][daysoy-(n-doy):daysoy]
                feature.extend(self.data[year][:doy])
            else:
                feature=self.data[year][doy-n:doy]
        except:
            feature=[nan]
        return feature

    def get_feature(self, decadal_date, n=1, threshold=0):
        # gives back timeseries of n decades length
        features=[]
        nans=[]
        try:
            while len(features)<n:
                value_sum=0
                m=0
                daydatelist=decadal_date.decade2days()
                for date in daydatelist:
                    value=self.get_dailyfeature(date, n=1)
                    if not isnan(value[-1]):
                        value_sum=value_sum+value[-1]
                        m=m+1
                features.append(value_sum)
                nans.append(1-float(m)/len(daydatelist))
                decadal_date=decadal_date.timedelta(-1)
        except:
            return [nan]
        if max(nans)>threshold or mean(nans)>threshold:
            return [nan]
        else:
            features.reverse()
            return features

    def update(self, date=None):
        start_date=self.method.first_date()
        end_date=self.method.last_date()
        if date==None:
            end_date=min(datetime.date.today(),end_date)
            start_date=max(self.last_observation(),start_date)
        elif self.method.last_date() > date > self.method.first_date():
            end_date=date
            start_date=date
        else:
            return 'date is out of range for this datasource'

        if start_date<=end_date:
            for date in self.daterange(start_date, end_date):
                #print 'update %s ...' %date.strftime("%Y-%m-%d")
                value=self.method.get_observation(self.database,date)
                doy=(date-datetime.date(date.year,1,1)).days+1
                if date.year>self.years[-1]:
                    self.years.append(date.year)
                    self.data.update({date.year: [float('nan')]*366})
                self.data[date.year][doy-1]=value
            output='database has been updated'
        else:
            output='database is already up-to date'
        return output

    def write(self):
        xlsxutils.write_data(self.data,join(self.database.filename,self.datatype+'.xlsx'))

    def daterange(self,start_date, end_date):
        for n in range(int ((end_date - start_date).days+1)):
            yield start_date + datetime.timedelta(n)

    def last_observation(self):
        lastyear=self.data[self.years[-1]]
        for i,e in reversed(list(enumerate(lastyear))):
            if isnan(lastyear[i]) and not isnan(lastyear[i-1]):
                return datetime.date(self.years[-1],1,1)+datetime.timedelta(i)
        if isnan(lastyear[-1]):
            return datetime.date(self.years[-1],1,1)
        else:
            return datetime.date(self.years[-1]+1,1,1)

    def leapyear(self,year):
        if year % 400 == 0:
            return True
        if year % 100 == 0:
            return False
        if year % 4 == 0:
            return True
        else:
            return False

    def transform2delta(self):
        datanew=dict()
        for year in self.years:
            datanew.update({year: [float('nan')] * 366})

        for date in self.daterange(datetime.date(self.years[0], 1,2), self.last_observation()+datetime.timedelta(-1)):
            doy=(date-datetime.date(date.year,1,1)).days+1
            date2=date+datetime.timedelta(-1)
            datanew[date.year][doy-1]=self.data[date.year][doy-1]-self.data[date2.year][(date2-datetime.date(date2.year,1,1)).days]
        return TransformedDailyDataset(self.datatype,self.years,datanew)

class TransformedDailyDataset(object):
    def __init__(self, datatype,years,data):
        self.datatype=datatype
        self.data = data
        self.years=years

    def get_dailyfeature(self, date, n=1):
        # gives back timeseries of n days length
        year=date.year
        doy=(date-datetime.date(year,1,1)).days+1
        #doy=(date.lastdate()-datetime.date(year,1,1)).days+1
        try:
            if doy <= n:
                daysoy = 366 if self.leapyear(year-1) else 365
                feature=self.data[year-1][daysoy-(n-doy):daysoy]
                feature.extend(self.data[year][:doy])
            else:
                feature=self.data[year][doy-n:doy]
        except:
            feature=[nan]
        return feature

    def get_feature(self, decadal_date, n=1, threshold=0):
        # gives back timeseries of n decades length
        features=[]
        nans=[]
        try:
            while len(features)<n:
                value_sum=0
                m=0
                daydatelist=decadal_date.decade2days()
                for date in daydatelist:
                    value=self.get_dailyfeature(date, n=1)
                    if not isnan(value[-1]):
                        value_sum=value_sum+value[-1]
                        m=m+1
                features.append(value_sum)
                nans.append(1-float(m)/len(daydatelist))
                decadal_date=decadal_date.timedelta(-1)
        except:
            return [nan]
        if max(nans)>threshold or mean(nans)>threshold:
            return [nan]
        else:
            features.reverse()
            return features

    def daterange(self,start_date, end_date):
        for n in range(int ((end_date - start_date).days+1)):
            yield start_date + datetime.timedelta(n)

    def last_observation(self):
        lastyear=self.data[self.years[-1]]
        for i,e in reversed(list(enumerate(lastyear))):
            if isnan(lastyear[i]) and not isnan(lastyear[i-1]):
                return datetime.date(self.years[-1],1,1)+datetime.timedelta(i)
        if isnan(lastyear[-1]):
            return datetime.date(self.years[-1],1,1)
        else:
            return datetime.date(self.years[-1]+1,1,1)

    def leapyear(self,year):
        if year % 400 == 0:
            return True
        if year % 100 == 0:
            return False
        if year % 4 == 0:
            return True
        else:
            return False

MAPPING = {
    'trmm': type_methods.precipitation,
    'sca':type_methods.precipitation,
}

if __name__ == '__main__':
    database=pkgutil.get_loader(join('database/chatkal'))
    dataset = DailyDataset('sca',database)
    dataset2=dataset.transform2delta()
    vec = dataset.get_dailyfeature(datetime.date(2010,12,31),365)
    vec2 = dataset2.get_dailyfeature(datetime.date(2010,12,31),365)
    import matplotlib.pyplot as plt
    plt.plot(vec)
    plt.plot(vec2)
    plt.draw()
    plt.savefig('test.png')
