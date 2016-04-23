from poster.encode import multipart_encode
from poster.streaminghttp import register_openers
import urllib2

register_openers()

with open("/home/jules/bla.csv", 'r') as f:
    datagen, headers = multipart_encode({"file": f})
    request = urllib2.Request("http://0.0.0.0:5000/upload/pskem", \
                              datagen, headers)
    response = urllib2.urlopen(request)