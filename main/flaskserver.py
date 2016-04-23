import json
from os.path import join, isdir
from os import makedirs, rmdir, remove, walk
from timeseriesFC import Forecaster, config2paramdict, database_loader
import pickle
from urllib2 import urlopen
from math import isnan
import os
from flask import Flask, request
from werkzeug import secure_filename
import datetime

ROOT_FOLDER='/home/jules/flaskserver/'
DATA_FOLDER = join(ROOT_FOLDER, 'data')
UPLOAD_FOLDER = DATA_FOLDER
ALLOWED_EXTENSIONS = set(['csv', 'json',])
app = Flask(__name__)
app.config.from_object(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def setup_app(app):
    if not isdir(ROOT_FOLDER):
        makedirs(ROOT_FOLDER)
    elif not isdir(DATA_FOLDER):
        makedirs(DATA_FOLDER)

setup_app(app)

@app.route('/')
def status():
    return message(True,None,ROOT_FOLDER)

@app.route('/forecast/<catchment>')
def forecast(catchment):
    database = database_loader(join(DATA_FOLDER, catchment))
    type=request.args.get('type','')
    date=request.args.get('date','')
    if date:
        try:
            if len(date) is not 8:
                raise Exception('Date string is too long or short')
            date=datetime.date(int(date[:4]),int(date[4:6]),int(date[6:]))
        except:
            return message(False,None,"Date must be given as YYYYMMDD, e.g. 20150810 for 10. August 2015.")
    else:
        date=datetime.date.today()
    if get_config(database,type) is None:
        return message(False,None,"This forecasttype is not found in the configuration file.")
    else:
        model = pickle.load(open(join(database.filename,type+'.p'), "rb"))
        data=model.single_targetset(date)
        if isnan(data):
            value=model.forecast(date)
        else:
            return message(False, data, "No forecast is needed. A value of %.0f has been measured" % data)

        if isnan(value):
            return message(False,None,"There is not enough data for a forecast. Please upload missing data for this forecast model")
        else:
            return message(True,value,"The forecasted value is %.0f" %value)

@app.route('/train/<catchment>')
def train_and_save(catchment):
    type = request.args.get('type', '')
    database = database_loader(join(DATA_FOLDER, catchment))
    if type:
        config=get_config(database,type)
    else:
        config=get_config(database)

    if config==None:
        return message(False,None,"Folder or configuration file for specified catchment was not found or the configuration File has no valid json syntax")

    if type:
        try:
            model = Forecaster(**config2paramdict(config, database))
            model.train_model()
        except:
            return message(False, None, "There was an Error while training. Check the configuration or data files for errors.")
        pickle.dump(model, open(join(database.filename, type + '.p'), "wb"))

    else:
        for descriptor in config.keys():
            try:
                model = Forecaster(**config2paramdict(config[descriptor], database))
                model.train_model()
            except:
                return message(False,None,"There was an Error while training. Check the configuration or data files for errors.")
            pickle.dump(model, open(join(database.filename,descriptor+'.p'), "wb"))

    return message(True,None,"The model has succesfully been trained")

@app.route('/crossvalidate/<catchment>')
def crossvalidate(catchment):
    type=request.args.get('type','')
    database = database_loader(join(DATA_FOLDER, catchment))
    config = get_config(database,type)
    if config == None:
        return message(False,None,"Folder or configuration file for specified catchment and type was not found")
    try:
        model = Forecaster(**config2paramdict(config, database))
        score=model.cross_validate()
    except:
        return message(False,None,"There was an Error. Check the configuration file for errors")
    return message(True,score,"The configured model reached a score of %0.2f" %score)

@app.route('/new/<catchment>')
def new(catchment):
    if not isdir(join(DATA_FOLDER, catchment)):
        makedirs(join(DATA_FOLDER, catchment))
        return message(True,None,"Catchment folder has been created.")
    else:
        return message(False,None,"Catchment folder already exists. Try uploading data.")

@app.route('/delete/<catchment>')
def delete(catchment):
    path=join(DATA_FOLDER, catchment)
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
    database = database_loader(join(DATA_FOLDER, catchment))
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = join(database.filename, filename)
            if os.path.exists(filepath):
                os.remove(filepath)
            file.save(join(database.filename, filename))
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

def get_config(database,descriptor=None):
    try:
        configfilepath=join(database.filename, 'config.json')
        with open(configfilepath) as json_config_file:
            config = json.load(json_config_file)
        if descriptor==None:
            return config
        else:
            if descriptor in config.keys():
                return config[descriptor]
            else:
                return None
    except:
        return None

def message(status,value,msg):
    return json.dumps({"status":status,"value":value,"message":msg})


if __name__ == '__main__':
    app.run(debug=True)

    ## ------------- Download forecast --------------
    #params={'type': 'monthly', 'date': '20101211'}
    #r = requests.get('http://0.0.0.0:5000/forecast/pskem',params=params)
    ## ------------- Upload file --------------------
    # files = {'file': open('/home/jules/bla.csv, 'rb')}
    #    r = requests.post('http://0.0.0.0:5000/upload/pskem', files=files)

