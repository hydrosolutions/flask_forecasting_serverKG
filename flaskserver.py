import datetime
import json
import os
import pickle
from math import isnan
from os import makedirs, rmdir, remove, walk
from os.path import join, isdir, realpath, dirname
from urllib2 import urlopen

from flask import Flask, request
from werkzeug import secure_filename

from timeseriesFC import Forecaster, config2paramdict, DatabaseLoader

DATA_FOLDER = join(dirname(realpath(__file__)),'sample_database/')
UPLOAD_FOLDER = DATA_FOLDER
ALLOWED_EXTENSIONS = {'csv', 'json'}
app = Flask(__name__)
app.config.from_object(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def setup_app(app):
    if not isdir(DATA_FOLDER):
        makedirs(DATA_FOLDER)

setup_app(app)

@app.route('/')
def status():
    return message(True, None, DATA_FOLDER)


@app.route('/forecast/<catchment>')
def forecast(catchment):
    database = DatabaseLoader(join(DATA_FOLDER, catchment))
    modeltype = request.args.get('type', '')
    date = request.args.get('date', '')
    if date:
        try:
            if len(date) is not 8:
                raise Exception('Date string is too long or short')
            date = datetime.date(int(date[:4]), int(date[4:6]), int(date[6:]))
        except:
            return message(False, None, "Date must be given as YYYYMMDD, e.g. 20150810 for 10. August 2015.")
    else:
        date = datetime.date.today()
    if get_config(database, modeltype) is None:
        return message(False, None, "This forecasttype is not found in the configuration file.")
    else:
        try:
            model = pickle.load(open(join(database.filename, modeltype + '.p'), "rb"))
        except:
            return message(False, None, "No trained model was found. Train model bedore forecasting")
        data = model.single_targetset(date)
        if isnan(data):
            value = model.forecast(date)
        else:
            return message(False, data, "No forecast is needed. A value of %.0f has been measured" % data)

        if isnan(value):
            return message(False, None,
                           "There is not enough data for a forecast. Please upload missing data for this "
                           "forecast model")
        else:
            return message(True, value, "The forecasted value is %.0f" % value)


@app.route('/train/<catchment>')
def train_and_save(catchment):
    modeltype = request.args.get('type', '')
    database = DatabaseLoader(join(DATA_FOLDER, catchment))
    if modeltype:
        config = get_config(database, modeltype)
    else:
        config = get_config(database)

    if config is None:
        return message(False, None,
                       "Folder or configuration file for specified catchment was not found or the configuration File "
                       "has no valid json syntax")

    if modeltype:
        try:
            model = Forecaster(**config2paramdict(config, database))
            model.train_model()
        except:
            return message(False, None,
                           "There was an Error while training. Check the configuration or data files for errors.")
        pickle.dump(model, open(join(database.filename, modeltype + '.p'), "wb"))

    else:
        for descriptor in config.keys():
            try:
                model = Forecaster(**config2paramdict(config[descriptor], database))
                model.train_model()
            except:
                return message(False, None,
                               "There was an Error while training. Check the configuration or data files for errors.")
            pickle.dump(model, open(join(database.filename, descriptor + '.p'), "wb"))

    return message(True, None, "The model has succesfully been trained")


@app.route('/crossvalidate/<catchment>')
def crossvalidate(catchment):
    modeltype = request.args.get('type', '')
    database = DatabaseLoader(join(DATA_FOLDER, catchment))
    config = get_config(database, modeltype)
    if config is None:
        return message(False, None, "Folder or configuration file for specified catchment and type was not found")
    try:
        model = Forecaster(**config2paramdict(config, database))
        score = model.cross_validate()
    except:
        return message(False, None, "There was an Error. Check the configuration file for errors")
    return message(True, score, "The configured model reached a score of %0.2f" % score)


@app.route('/new/<catchment>')
def new(catchment):
    if not isdir(join(DATA_FOLDER, catchment)):
        makedirs(join(DATA_FOLDER, catchment))
        return message(True, None, "Catchment folder has been created.")
    else:
        return message(False, None, "Catchment folder already exists. Try uploading data.")


@app.route('/delete/<catchment>')
def delete(catchment):
    path = join(DATA_FOLDER, catchment)
    if isdir(path):
        for root, dirs, files in walk(path, topdown=False):
            print root
            print dirs
            print files
            for name in files:
                remove(join(root, name))
            for name in dirs:
                rmdir(join(root, name))
        rmdir(path)
        return message(True, None, "Catchment folder has been deleted.")
    else:
        return message(False, None, "Catchment could not be deleted. There might be a problem with filesystem rights.")


@app.route('/upload/<catchment>', methods=['GET', 'POST'])
def upload_file(catchment):
    database = DatabaseLoader(join(DATA_FOLDER, catchment))
    if request.method == 'POST':
        fileobject = request.files['file']
        if fileobject and allowed_file(fileobject.filename):
            filename = secure_filename(fileobject.filename)
            filepath = join(database.filename, filename)
            if os.path.exists(filepath):
                os.remove(filepath)
            fileobject.save(join(database.filename, filename))
            return message(True, None, "File has been uploaded")
        else:
            return message(False, None, "File could not be uploaded")
    if request.method == 'GET':
        url = request.args.get('url', '')
        filename = request.args.get('filename', '')
        if url and allowed_file(filename):
            filename = secure_filename(filename)
            filepath = join(database.filename, filename)
            if os.path.exists(filepath):
                os.remove(filepath)

            f = urlopen(url)
            local_file = open(filepath, "wb")
            local_file.write(f.read())
            local_file.close()
            return message(True, None, "File has been uploaded")
        else:
            return message(False, None, "File could not be uploaded")
    else:
        return message(False, None, "File could not be uploaded")


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


def get_config(database, descriptor=None):
    try:
        configfilepath = join(database.filename, 'config.json')
        with open(configfilepath) as json_config_file:
            config = json.load(json_config_file)
        if descriptor is None:
            return config
        else:
            if descriptor in config.keys():
                return config[descriptor]
            else:
                return None
    except:
        return None


def message(statusbool, value, msg):
    return json.dumps({"status": statusbool, "value": value, "message": msg})


if __name__ == '__main__':
    app.run(debug=True)
