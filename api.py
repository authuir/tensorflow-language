#!/usr/bin/python
#coding=utf-8

import web
import json
import subprocess
from vietnamese_CNN import predic

urls = (
    '/api', 'Api',
    )
app = web.application(urls,globals())

class Api:
    def POST(self):  
        i = web.input().data
        data = json.loads(i)
        return json.dumps(predic(data))

if __name__ == "__main__":
    web.wsgi.runwsgi = lambda func, addr=None: web.wsgi.runfcgi(func, addr)
    app.run()
