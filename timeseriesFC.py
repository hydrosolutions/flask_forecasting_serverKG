import csv
import datetime
from os import path, remove

import numpy as np
from sklearn.cross_validation import StratifiedKFold
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.metrics import r2_score

from utils.datawrapper import Dataset
from utils.tools import DatabaseLoader
from utils.special_datetime import DecadalDate as DateFormat


class Forecaster(object):
    def __init__(self, modeltype, datasets, leadtimes, mode='default', fillnan=False, scoremethod='R2'):
        self.datasets = datasets
        self.model = modeltype
        self.lead_times = self.check_lead_time_order(leadtimes)
        if mode == 'naiveaverage':
            self.forecast = self.forecastwithaverage
        elif mode == 'naivelast':
            self.forecast = self.forecastwithlast
        self.fillnan = fillnan
        if fillnan:
            self.threshold = 0.075
        else:
            self.threshold = 0
        self.scoremethod = scoremethod

    def cross_validate(self, daterange=None, folds=10, output=None):
        if output is None:
            output = self.scoremethod

        if daterange is None:
            miny = 9999
            maxy = 0
            for dataset in self.datasets:
                if dataset.years[0] < miny:
                    miny = dataset.years[0]
                if dataset.years[-1] > maxy:
                    maxy = dataset.years[-1]
            startyear = DateFormat(miny, 1)
            endyear = DateFormat(maxy + 1, 1)
            daterange = DateFormat.decadal_daterange(startyear, endyear)

        targetset, featureset, datelist = self.cleanup_daterange(daterange)
        decades = []
        for date in datelist:
            decades.append(date.decade_of_year)

        kf = StratifiedKFold(decades, folds)
        score = []
        for train_index, test_index in kf:
            self.model.fit(featureset[train_index], targetset[train_index])
            score.append(self.evaluate(datelist[test_index], output))
        return np.nanmean(score)

    def cleanup_daterange(self, daterange):

        feature_length = self.lead_time2length(self.lead_times[1:])
        featureset = np.zeros([len(daterange), feature_length])
        targetset = np.zeros([len(daterange), 1])
        complete_dates = np.ones([len(daterange), 1], bool)

        # Create featureset
        for j, date in enumerate(daterange):
            feature = self.single_featureset(date)
            target = self.single_targetset(date)
            if not (any(np.isnan(feature)) or np.isnan(target)):
                featureset[j, :] = feature
                targetset[j] = target
            else:
                complete_dates[j] = False

        # Clean up Featureset and Targetset
        numberofvaliddates = np.sum(complete_dates)
        cleaned_featureset = np.zeros([numberofvaliddates, feature_length])
        cleaned_targetset = np.zeros(numberofvaliddates)
        cleaned_training_dates = []
        m = 0
        for i, bool_value in enumerate(complete_dates):
            if bool_value:
                cleaned_featureset[m] = featureset[i]
                cleaned_targetset[m] = targetset[i]
                cleaned_training_dates.append(daterange[i])
                m += 1

        return cleaned_targetset, cleaned_featureset, np.array(cleaned_training_dates)

    @staticmethod
    def check_lead_time_order(lead_times):
        for n in lead_times:
            if n[1] < n[0]:
                temp = n[0]
                n[0] = n[1]
                n[1] = temp
        return lead_times

    def shortterm_validation(self):
        if self.lead_times[0][1] == 0 and self.lead_times[0][0] == 0:
            return True
        else:
            return False

    @staticmethod
    def lead_time2length(lead_times):
        """ Return sum of lengths of one or more leadtime intervals as single value.

        e.g. lead_time2length([[-3,-1],[1,10],[1,1]) returns 14
        """
        sumvalue = 0
        for n in lead_times:
            sumvalue = sumvalue + n[1] - n[0] + 1
        return sumvalue

    def single_featureset(self, date):
        """ Collect featureset for specified date from datasets and lead_times.

        First entry in datasets resp. lead_times is ignored, as it is defined as targetset.
        Return list of values.
        Returns NaN if one or more values in datasets are missing for that date.
        """
        lead_times = self.lead_times[1:]
        n = self.lead_time2length(lead_times)
        singlefeatureset = np.zeros(n)
        position = 0
        for i, dataset in enumerate(self.datasets[1:]):
            length = self.lead_time2length([lead_times[i]])
            timeseries = dataset.get_feature(date.timedelta(lead_times[i][1]), length, self.threshold, self.fillnan)
            if timeseries is not np.nan:
                singlefeatureset[position:position + length] = timeseries
            else:
                return np.nan
            position += length
        return singlefeatureset

    def single_targetset(self, date):
        """ Return targetvalue for specified date from first entry in datasets.

        Returns average value over lead_times specified for targetset -> lead_times[0]
        If any value is missing, return NaN
        """
        if isinstance(date, datetime.date):
            date = DateFormat.datetime2date(date)

        dataset = self.datasets[0]
        forecasting_leadtime = self.lead_times[0]
        targets = []
        for dT in range(forecasting_leadtime[0], forecasting_leadtime[1] + 1):
            targets.append(dataset.get_feature(date.timedelta(dT))[0])
        if any(np.isnan(targets)):
            return np.nan
        else:
            return np.mean(targets)

    def train_model(self, training_daterange=None):
        if training_daterange is None:
            miny = 9999
            maxy = 0
            for dataset in self.datasets:
                if dataset.years[0] < miny:
                    miny = dataset.years[0]
                if dataset.years[-1] > maxy:
                    maxy = dataset.years[-1]
            startyear = DateFormat(miny, 1)
            endyear = DateFormat(maxy + 1, 1)
            training_daterange = DateFormat.decadal_daterange(startyear, endyear)

        targetset, featureset, datelist = self.cleanup_daterange(training_daterange)
        self.model.fit(featureset, targetset)
        return None

    def forecast(self, date):
        """ Wrapper for Skilearn predict method. Takes date as argument

        Returns predicted value. If featureset is incomplete, returns NaN.
        """
        if isinstance(date, datetime.date):
            date = DateFormat.datetime2date(date)

        feature = self.single_featureset(date)
        if not any(np.isnan(feature)):
            feature2 = feature.reshape(1, feature.shape[0])
            return self.model.predict(feature2)[0]
        else:
            return np.nan

    def forecastwithaverage(self, date):
        """ Naive forecaster: Forecast is average value for this time of the year"""
        decades = []
        for date2 in DateFormat.decadal_daterange(date.timedelta(self.lead_times[0][0]),
                                                  date.timedelta(self.lead_times[0][1])):
            decades.append(date2.decade_of_year)
        return self.datasets[0].decadal_average(decades)

    def forecastwithlast(self, date):
        """ Naive forecaster: Forecast is last known value"""
        return self.datasets[0].get_feature(date.timedelta(-1), 1)[0]

    def evaluate(self, daterange, output=None):
        if output is None:
            output = self.scoremethod

        predicted = np.empty(len(daterange))
        average = np.empty(len(daterange))
        observed = np.empty(len(daterange))
        datelist = np.full(len(daterange), None, dtype=DateFormat)
        for i, date in enumerate(daterange):
            predicted[i] = self.forecast(date)
            average[i] = self.forecastwithaverage(date)
            observed[i] = self.single_targetset(date)
            datelist[i] = date.firstdate()

        if output == 'R2':
            mask = np.logical_or(np.isnan(predicted), np.isnan(observed))
            return r2_score(observed[~mask], predicted[~mask])

        elif output == 'soviet_longterm':
            stdev = []
            for date in daterange:
                decades = []
                for date2 in DateFormat.decadal_daterange(date.timedelta(self.lead_times[0][0]),
                                                          date.timedelta(self.lead_times[0][1])):
                    decades.append(date2.decade_of_year)
                stdev.append(self.datasets[0].decadal_standard_deviation(decades))

            stdev = np.array(stdev, dtype=np.float)
            error = np.absolute(np.array(observed, dtype=np.float) - np.array(predicted, dtype=np.float))
            nonnans = ~np.isnan(error)
            return np.mean(error[nonnans] / stdev[nonnans])

        elif output == 'soviet_shortterm':
            if self.shortterm_validation():
                stdev = np.empty(len(daterange))
                targetdiff = self.datasets[0].transform2delta()
                for i, date in enumerate(daterange):
                    stdev[i] = targetdiff.decadal_standard_deviation([date.decade_of_year])

                stdev = np.array(stdev, dtype=np.float)
                error = np.absolute(np.array(observed, dtype=np.float) - np.array(predicted, dtype=np.float))
                nonnans = ~np.isnan(error)
                return np.mean(error[nonnans] / stdev[nonnans])
            else:
                print 'shortterm evaluator is onyl valid for target leadtime [0,0]'
                return None

        else:
            print 'score method can be: R2,soviet_longterm,soviet_shortterm'
            return np.nan

    def plot(self, daterange, output=None, filename='evaluate.png'):
        if output is None:
            output = self.scoremethod

        predicted = np.empty(len(daterange))
        average = np.empty(len(daterange))
        observed = np.empty(len(daterange))
        datelist = np.full(len(daterange), None, dtype=DateFormat)
        for i, date in enumerate(daterange):
            predicted[i] = self.forecast(date)
            average[i] = self.forecastwithaverage(date)
            observed[i] = self.single_targetset(date)
            datelist[i] = date.firstdate()

        if output == 'timeseries':
            from matplotlib import pyplot as plt
            plt.figure(figsize=(15, 5))
            # plt.figure(figsize=(7,5))

            plt.plot(datelist, predicted, label='predicted', color='red')
            plt.plot(datelist, average, label='historical average', linestyle='--', linewidth=0.5, color='green')
            plt.plot(datelist, observed, label='observed', color='blue')
            plt.ylabel(str(self.datasets[0].datatype))
            plt.xticks(rotation=90)
            plt.tight_layout()
            # plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.legend()
            plt.draw()
            plt.savefig(filename)
            return None

        elif output == 'correlation':
            from matplotlib import pyplot as plt
            plt.figure(figsize=(5, 5))
            plt.plot(observed, predicted, linestyle='None', marker='.')
            maxval = int(np.nanmax([np.max(observed), np.nanmax(predicted)]))
            minval = int(np.min([np.nanmin(observed), np.nanmin(predicted)]))
            plt.xlim((minval * 0.9, maxval * 1.1))
            plt.ylim((minval * 0.9, maxval * 1.1))
            plt.plot(plt.gca().get_xlim(), plt.gca().get_ylim(), ls="--", c=".3")
            plt.xlabel('observed ' + str(self.datasets[0].datatype))
            plt.ylabel('forecasted ' + str(self.datasets[0].datatype))
            plt.draw()
            plt.savefig(filename)
            return None

        elif output == 'soviet_longterm':

            # Does not account for unsorted dateranges!
            years = range(daterange[0].year, daterange[-1].year + 1)

            dates = range(1, daterange[0].datesperyear() + 1)
            nbofdates = dates[-1]
            error = np.full([len(years), nbofdates], np.nan)
            stdev = np.full(nbofdates, np.nan)
            axis = np.full(nbofdates, np.nan)

            for date in daterange:
                relativeyear = date.year - daterange[0].year
                decades = []
                for date2 in DateFormat.decadal_daterange(date.timedelta(self.lead_times[0][0]),
                                                          date.timedelta(self.lead_times[0][1])):
                    decades.append(date2.decade_of_year)
                error[relativeyear, date.decade_of_year - 1] = np.absolute(
                    self.forecast(date) - self.single_targetset(date)) / (
                                                                   datasets[0].decadal_standard_deviation(decades))
                axis[date.decade_of_year - 1] = date.decade_of_year

            error2 = np.ma.masked_invalid(error)
            error = [[y for y in row if y] for row in error2.T]

            from matplotlib import pyplot as plt

            plt.figure(figsize=(15, 5))
            plt.boxplot(error, positions=dates, showmeans=True)
            plt.plot(axis, stdev, label='standard deviation')
            plt.axhline(0.8, linestyle='--', color='b', label='80% of standard deviation')
            plt.axhline(0.6, linestyle='--', color='g', label='60% of standard deviation')
            axes = plt.gca()
            plt.xlabel('issue date (decade of year)')
            plt.ylabel('error/STDEV')
            axes.set_ylim([0, 1.5])
            plt.legend()
            plt.draw()
            plt.savefig(filename)
            return None

        elif output == 'soviet_shortterm':

            if self.shortterm_validation():
                targetdiff = datasets[0].transform2delta()
                years = range(daterange[0].year, daterange[-1].year + 1)
                dates = range(1, daterange[0].datesperyear() + 1)
                nbofdates = dates[-1]
                error = np.full([len(years), nbofdates], np.nan)
                stdev = np.full(nbofdates, np.nan)
                axis = np.full(nbofdates, np.nan)

                for i, date in enumerate(daterange):
                    relativeyear = date.year - daterange[0].year
                    error[relativeyear, date.decade_of_year - 1] = np.absolute(
                        predicted[i] - observed[i]) / targetdiff.decadal_standard_deviation([date.decade_of_year])

                error2 = np.ma.masked_invalid(error)
                error = [[y for y in row if y] for row in error2.T]

                from matplotlib import pyplot as plt

                plt.figure(figsize=(15, 5))
                plt.plot(axis, stdev, label='standard deviation')
                plt.axhline(0.674, linestyle='--', color='g', label='67.4% of standard deviation')
                plt.boxplot(error, positions=dates, showmeans=True)
                axes = plt.gca()
                axes.set_ylim([0, 1.5])
                plt.ylabel('error/STDEV')
                plt.xlabel('issue date (decade of year)')
                plt.legend()
                plt.draw()
                plt.savefig(filename)
                return None
            else:
                print 'shortterm evaluator is onyl valid for target leadtime [0,0]'
                return None

        elif output == 'importance':
            vec = self.model.feature_importances_
            importance = []
            data = []
            tailtimes = []

            j = 0
            for i, dataset in enumerate(self.datasets[1:]):
                length = self.lead_time2length([self.lead_times[i + 1]])
                tailtimes.append(range(self.lead_times[i + 1][0], self.lead_times[i + 1][1] + 1))
                data.append(dataset.datatype)
                importance.append(vecpart)
                j += length

            from matplotlib import pyplot as plt
            plt.figure(figsize=(10, 5))
            for i, vec in enumerate(importance):
                plt.plot(tailtimes[i], vec, label=data[i])
            plt.legend(loc='upper left')
            plt.yscale('log')
            plt.xlabel('tailtime')
            plt.ylabel('importance [-]')
            plt.draw()
            plt.savefig(filename)
            return None

    def csv(self, daterange, filename='result.csv'):
        if path.isfile(filename):
            remove(filename)
        writer = csv.writer(open(filename, 'wb'), delimiter=',')

        label = ['year', 'decade', 'observed', 'forecast', 'historic average', 'STDEV Q', 'STDEV deltaQ']
        writer.writerow(label)

        targetset = self.datasets[0]
        targetdiff = targetset.transform2delta()
        for i, date in enumerate(daterange):
            decades = []
            for date2 in DateFormat.decadal_daterange(date.timedelta(self.lead_times[0][0]),
                                                      date.timedelta(self.lead_times[0][1])):
                decades.append(date2.decade_of_year)
            data = [date.year, date.decade_of_year, self.single_targetset(date), self.forecast(date),
                    self.forecastwithaverage(date), targetset.decadal_standard_deviation(decades),
                    targetdiff.decadal_standard_deviation(decades)]
            writer.writerow(data)
        return None


def config2paramdict(config, database):
    params = ['modeltype', 'modelparam', 'datasets', 'leadtimes', 'scoremethod', 'fillnan']
    for namestr in params:
        assert namestr in config.keys(), 'the parameter "' + namestr + '" has no entry in JSON config File'

    if config['modeltype'] == 'RF':
        modeltype = RandomForestRegressor(**config['modelparam'])
    elif config['modeltype'] == 'EF':
        modeltype = ExtraTreesRegressor(**config['modelparam'])
    else:
        print 'modeltype not supported'
        return None

    datasets_in = []
    for datasettype in config['datasets']:
        datasets_in.append(Dataset(datasettype, database))

    paramdict = {'modeltype': modeltype,
                 'scoremethod': config['scoremethod'],
                 'fillnan': config['fillnan'],
                 'datasets': datasets_in,
                 'leadtimes': config['leadtimes']
                 }
    return paramdict


if __name__ == '__main__':

    # Select database (catchment)
    database = DatabaseLoader(
        '/home/jules/Dropbox (hydrosolutions)/Hydromet/Forecasting_Dev/Forecasting_KG/sample_database/chatkal')

    # Select target and feature dataset(s) --> [target, feature1, feature2, ... ]
    datasets = [Dataset('runoff', database), Dataset('runoff', database), Dataset('temp', database),
                Dataset('precip', database)]

    # Select leadtimes for target and feature. negative:past/positive:future
    leadtimes = [[1, 3], [-24, -1], [-24, -1], [-24, -1]]

    # Select Model
    # model_type=Lasso(alpha=1)

    # from sknn.mlp import Regressor, Layer, MultiLayerPerceptron
    # model_type=Regressor(layers=[Layer("Sigmoid", units=50),Layer("Linear")],learning_rate=0.1,n_iter=100)
    model_type = RandomForestRegressor(n_estimators=100, bootstrap=True, min_weight_fraction_leaf=0, max_depth=None)

    # Set training interval
    startyear = DateFormat(1933, 1)
    endyear = DateFormat(2015, 36)
    training_daterange = DateFormat.decadal_daterange(startyear, endyear)

    # Set testing interval
    startyear = DateFormat(2015, 1)
    endyear = DateFormat(2015, 1)
    testing_daterange = DateFormat.decadal_daterange(startyear, endyear)
    newtesting_daterange = []
    for date in testing_daterange:
        if date.decade_of_year > 0:  # Selecting last decade of each month as issue date
            newtesting_daterange.append(date)

    # Creates forecasting model with selected parameters
    model = Forecaster(model_type, datasets, leadtimes, fillnan=True, scoremethod='soviet_longterm')
    model.train_model(training_daterange)
    print model.forecast(DateFormat(2015, 1))
    # print model.plot(newtesting_daterange,output='timeseries')

    """## EXTENDED GRIDSEARCH FOR PARAMETERES AND DATASETS
    gridsearch=[]
    import itertools

    RF_options = {'n_estimators': [100]}

    product = [x for x in apply(itertools.product, RF_options.values())]
    RF_list = [dict(zip(RF_options.keys(), p)) for p in product]

    target=['runoff']
    target_leadtime=[[0,0]]

    available_datasets=['runoff','temp','dsca','sca']
    DATA_list=[c for i in range(len(available_datasets)) for c in itertools.combinations(available_datasets, i+1)]

    possible_leadtimes=[[-24,-1]]

    for RF_option in RF_list:
        for DATA in DATA_list:
            n=len(DATA)
            leadtime_list=list(itertools.product(possible_leadtimes,repeat=n))
            for leadtimes in leadtime_list:
                settings=dict(RF_option)
                settings['DATA']=list(DATA)
                settings['lead_times']=list(leadtimes)
                gridsearch.append(settings)
                model_type = RandomForestRegressor(**RF_option)

                datasets=target+list(DATA)
                leadtimeset=target_leadtime+list(leadtimes)
                data_in=[]
                for type in datasets:
                    if type=='sca':
                        data_in.append(DailyDataset(type, database))
                    elif type=='dsca':
                        data_in.append(DailyDataset('sca', database).transform2delta())
                    else:
                        data_in.append(Dataset(type, database))

                model=Forecaster(model_type,data_in,leadtimeset)

                from timeit import default_timer as timer
                start = timer()
                score=model.cross_validate(training_daterange,'soviet_shortterm')
                end = timer()
                gridsearch[-1]['elapsed time']=end - start
                gridsearch[-1]['score']=score
                print gridsearch[-1]



    import xlsxwriter

    workbook = xlsxwriter.Workbook('gridsearch_result.xlsx')
    worksheet = workbook.add_worksheet()
    row = 0
    col = 0
    for key in gridsearch[0].keys():
        worksheet.write(row, col,  str(key))
        col += 1

    col=0
    row += 1

    for result in gridsearch:
        for key in result.keys():
            worksheet.write(row, col,  str(result[key]))
            col += 1
        col=0
        row += 1

    workbook.close()"""
