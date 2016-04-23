Last Change: 2016/04/02, jh

GENERAL INFORMATION

These Python Scripts run under every operating system, provided that Python 2.7 and all required libraries are installed on the computer. Under Linux, 
the most easy way to run them, is to activate the virtualenv environment in the folder /venv, which contains all required dependencies. 

DESCRIPTION

--> main --> timeseriesFC.py

contains the Class Forecaster, which is the toplayer forecasting module. It wraps the Skilearn machine learning methods in order
to work with the Class Dataset (here from feature_toolbox/decadal_dataset, more info see below)
The Forecaster Class can be initialized through defining the following Arguments:
	
	- modeltype: instance of a Skilearn machine learning class, e.g. RandomForestRegressor(n_estimators=100)
	- datasets: a list of Dataset instances, where the first one (index=0) is the targetdataset, the others are features
	- leadtimes: a list of lead- resp. tailtimes in the same order as the datasets. 
	  A tailtime is defined as 2-element list: [start,end]. The unit of start&end is the timestep of data in the 
	  corresponding Dataset (1 decade: decadal Dataset). The value can be negative (feature) or positive (target)
	  0 is supposed to be the time when the forecast is made. 
	  Example: feature tailtime: [-3,-1] , target leadtime: [1] , decadal Dataset
			   If the forecast is done today, the value of the next decade is forecasted, based on values of the three 
			   decades before the current one.
			   
The Forecaster has the following function. All other functions or supposed to be used by the class itself only.

	- Forecaster.crossvalidate(daterange,scoremethod,folds)
		- It returns the crossvalidated score of the defined forecasting model.
		- daterange: a list of dates in the format of the targetset Dataset.
		- scoremethod is one of the following: R2,soviet_longterm,soviet_shortterm, default='R2'
		- folds: number of cross validation folds, default=10
		
	- Forecaster.train_model(self, training_daterange)
		- trains the model over the given list of dates. If nor argument is given, all available data are used for training.
		- daterange: a list of dates in the format of the targetset Dataset.
	
	- Forecaster.forecast(date)
		- gives the forecast of the trained model for the given date.
		- date: a date in the format of the targetset Dataset. Corresponds to "0" in the lead- resp. tailtime definition.
		
	- Forecaster.evaluate(daterange, output):
		- returns the Score of the trained model over the specified daterange.
		- daterange: a list of dates in the format of the targetset Dataset.
		- output is one of the following: R2,soviet_longterm,soviet_shortterm, default='R2'

	- Forecaster.plot(daterange, output, filename):
		- Plots several informative graphs of the forecasting results.
		- daterange: a list of dates in the format of the targetset Dataset.
		- output=timeseries: simple value plot of the forecasted, observed and average values of the target Dataset
		- output=correlation: a scatter plot of forecasted and observed values
		- output=soviet_longterm: the soviet_longterm scoring results plotted for every timestep of a year
		- output=soviet_shortterm: the same as above but with the scoring method for shortterm forecasts
		- output=importance: plots the importance of each feature Dataset over its tailtime
		- filename: path to the imagefile to be created, default='evaluate.png'
		
		
--> feature_toolbox --> datawrapper.py

contains the class Dataset, which can handle timeseries datasets of different time resolutions. 
A Dataset instance is initialized with a datatype string and a database object. The latter is an instance of the database_loader class
in utils/tools.py. The database objects contains the path to the folder with the .csv timeseries files. The datatype string is the name 
of the corresponding csv file.

.....MORE TO COME......







		
	  
