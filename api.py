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
        rate, code = predic(data)
        rtn = '["%f","%f","%f","%s","%s","%s","%f","%f","%f","%s","%s","%s"]' % ( rate[0],rate[1],rate[2],code[0],code[1],code[2],rate[0],rate[1],rate[2],code[0],code[1],code[2])
        return rtn

if __name__ == "__main__":
    web.wsgi.runwsgi = lambda func, addr=None: web.wsgi.runfcgi(func, addr)
    app.run()
