from math import isnan, sqrt, pow
from os.path import join

import numpy as np

from utils import csvutils
from utils.special_datetime import DecadalDate as DateFormat


class Dataset(object):
    """ Dataset Class which can handle different time formats (Decades, more to come...)

    Initialise with name of Datatype and Database
    """

    def __init__(self, datatype, databaseobject):
        self.database = databaseobject
        assert self.database is not None, 'Folder for specifed station does not exist!'
        self.datatype = datatype
        self.data = self.__read_data()
        self.years = sorted(list(self.data.viewkeys()))

    def __read_data(self):
        try:
            return csvutils.load_data(join(self.database.filename, self.datatype + '.csv'))
        except:
            print "Dataset not found, create new"
            return None

    # OLD CODE SNIPPET FOR WORKING WITH REMOTE SENSED DATA. NOT NEEDED IN HYDROMET APPLICATION
    # def __create_empty(self):
    #         first_date=self.method.first_date()
    #         return {first_date.year: [float('nan')] * dateformat.datesperyear()}

    def get_feature(self, date, n=1, threshold=0.0):
        """ Return timeseries as list from date backward in time for n time steps"""
        features = []
        nan = 0
        try:
            while len(features) < n:
                decadeofyear = date.decade_of_year
                year = date.year
                value = self.data[year][decadeofyear - 1]
                if isnan(value):
                    nan += 1
                features.append(value)
                date = date.timedelta(-1)
        except:
            return [np.nan]
        features.reverse()
        features = np.array(features)
        if float(nan) / n > threshold:
            return [np.nan]
        elif nan > 0:
                nans, x = self.nan_helper(features)
                features[nans] = np.interp(x(nans), x(~nans), features[~nans])
        return features

    @staticmethod
    def nan_helper(x):
        return np.isnan(x), lambda z: z.nonzero()[0]

    def decadal_standard_deviation(self, decades_of_year):
        xi = []
        for year in self.years:
            values = []
            for decade in decades_of_year:
                values.append(self.data[year][decade - 1])
            if any(np.isnan(values)):
                average = np.nan
            else:
                average = np.mean(values)
            if not isnan(average):
                xi.append(average)

        xm = self.decadal_average(decades_of_year)

        std_dev = 0
        for x in xi:
            std_dev += pow((x - xm), 2)
        std_dev = sqrt(std_dev / (len(xi) - 1))
        return std_dev

    def decadal_average(self, decades_of_year):
        values = []
        for decade in decades_of_year:
            for year in self.years:
                values.append(self.data[year][decade - 1])
        return np.nanmean(values)

    # OLD CODE SNIPPET FOR WORKING WITH REMOTE SENSING DATA. NOT NEEDED HYDROMET APPLICATION
    # def update(self, date=None):
    #     ###### adapt date handling!!
    #     start_date=self.method.first_date()
    #     end_date=self.method.last_date()
    #     if date==None:
    #         end_date=dateformat.min(dateformat.today().timedelta(-1), end_date)
    #         start_date=dateformat.max(self.last_observation(), start_date)
    #     elif dateformat.decadal_difference(self.method.last_date(), date)>=0 and dateformat.
    #                                                       decadal_difference(self.method.first_date(), date)<=0:
    #         end_date=date
    #         start_date=date
    #     else:
    #         return 'date is out of range for this datasource'
    #
    #     if dateformat.decadal_difference(start_date, end_date)<=0:
    #         for date in self.decadal_daterange(start_date, end_date):
    #             #print date.year,date.decade_of_year
    #             value=self.method.get_observation(self.database,date)
    #             if date.year>self.years[-1]:
    #                 self.years.append(date.year)
    #                 self.data.update({date.year: [float('nan')] * dateformat.datesperyear()})
    #             self.data[date.year][date.decade_of_year-1]=value
    #         message='database has been updated'
    #     else:
    #         message='database is already up-to date'
    #     return message
    #
    # def write(self):
    #     csvutils.write_decadal_data(self.data, join(self.database.filename, self.datatype + '.xlsx'))

    @staticmethod
    def decadal_daterange(start_date, end_date):
        dates = []
        for n in range(int(DateFormat.decadal_difference(end_date, start_date) + 1)):
            dates.append(start_date.timedelta(n))
        return dates

    def last_observation(self):
        lastyear = self.data[self.years[-1]]
        for i, e in reversed(list(enumerate(lastyear))):
            if isnan(lastyear[i]) and not isnan(lastyear[i - 1]):
                return DateFormat(self.years[-1], i + 1)
        if isnan(lastyear[-1]):
            return DateFormat(self.years[-1], 1)
        else:
            return DateFormat(self.years[-1] + 1, 1)

    def max(self):
        maxvalue = -9999
        for date in DateFormat.decadal_daterange(DateFormat(self.years[0], 1), self.last_observation()):
            if self.data[date.year][date.decade_of_year - 1] > maxvalue:
                maxvalue = self.data[date.year][date.decade_of_year - 1]
        return maxvalue

    def transform2delta(self):
        datanew = dict()
        for year in self.years:
            datanew.update({year: [float('nan')] * DateFormat.datesperyear()})

        for date in DateFormat.decadal_daterange(DateFormat(self.years[0], 2), self.last_observation()):
            datanew[date.year][date.decade_of_year - 1] = self.data[date.year][date.decade_of_year - 1] - \
                                                          self.data[date.timedelta(-1).year][
                                                              date.timedelta(-1).decade_of_year - 1]
        return TransformedDataset(self.datatype, self.years, datanew)

    def normalized(self):
        maxvalue = self.max()
        datanew = dict()
        for year in self.years:
            datanew.update({year: [float('nan')] * DateFormat.datesperyear()})

        for date in DateFormat.decadal_daterange(DateFormat(self.years[0], 1), self.last_observation()):
            datanew[date.year][date.decade_of_year - 1] = self.data[date.year][date.decade_of_year - 1] / maxvalue
        return TransformedDataset(self.datatype, self.years, datanew)


class TransformedDataset(object):
    def __init__(self, datatype, years, datadict):
        self.datatype = datatype
        self.data = datadict
        self.years = years

    def get_feature(self, date, n=1, threshold=0.0):
        # gives back timeseries of n decades length
        features = []
        nan = 0
        try:
            while len(features) < n:
                decadeofyear = date.decade_of_year
                year = date.year
                value = self.data[year][decadeofyear - 1]
                if isnan(value):
                    nan += 1
                features.append(value)
                date = date.timedelta(-1)
        except:
            return [np.nan]
        features.reverse()
        features = np.array(features)
        if float(nan) / n > threshold:
            return [np.nan]
        elif nan > 0:
            nans, x = self.nan_helper(features)
            features[nans] = np.interp(x(nans), x(~nans), features[~nans])
        return features

    @staticmethod
    def nan_helper(x):
        return np.isnan(x), lambda z: z.nonzero()[0]

    def decadal_standard_deviation(self, decades_of_year):
        xi = []
        for year in self.years:
            values = []
            for decade in decades_of_year:
                values.append(self.data[year][decade - 1])
            if any(np.isnan(values)):
                average = np.nan
            else:
                average = np.mean(values)
            if not isnan(average):
                xi.append(average)

        xm = self.decadal_average(decades_of_year)

        std_dev = 0
        for x in xi:
            std_dev += pow((x - xm), 2)
        std_dev = sqrt(std_dev / (len(xi) - 1))
        return std_dev

    def decadal_average(self, decades_of_year):
        values = []
        for decade in decades_of_year:
            for year in self.years:
                values.append(self.data[year][decade - 1])
        return np.nanmean(values)

    @staticmethod
    def decadal_daterange(start_date, end_date):
        dates = []
        for n in range(int(DateFormat.decadal_difference(end_date, start_date) + 1)):
            dates.append(start_date.timedelta(n))
        return dates

    def last_observation(self):
        lastyear = self.data[self.years[-1]]
        for i, e in reversed(list(enumerate(lastyear))):
            if isnan(lastyear[i]) and not isnan(lastyear[i - 1]):
                return DateFormat(self.years[-1], i + 1)
        if isnan(lastyear[-1]):
            return DateFormat(self.years[-1], 1)
        else:
            return DateFormat(self.years[-1] + 1, 1)


if __name__ == '__main__':
    print None