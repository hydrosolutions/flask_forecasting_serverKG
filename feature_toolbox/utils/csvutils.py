import csv
from numpy import nan
from os import remove, path

def load_data(filepath):
    assert path.isfile(filepath), filepath+' is not a file!'
    reader = csv.reader(open(filepath, 'r'))
    d = {}
    for row in reader:
        intlist=[]
        for stringvalue in row[1:]:
            try:
                intlist.append(float(stringvalue))
            except:
                intlist.append(nan)
        d[int(row[0])]=intlist
    return d

def write_data(data,filepath):
    if path.isfile(filepath):
        remove(filepath)
    writer = csv.writer(open(filepath, 'wb'),delimiter =',')
    years = sorted(list(data.viewkeys()))
    for key in years:
        writer.writerow([key]+data[key])


