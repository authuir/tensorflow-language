#!/usr/bin/python
#coding=utf-8

import web
import json
import subprocess
from vietnamese_CNN import predic
import vietnamese

urls = (
    '/api', 'Api',
    )
app = web.application(urls,globals())

class Api:
    def POST(self):  
        i = web.input().data
        data = json.loads(i)
        rate, code = predic(data)
        rate1, code1 = vietnamese.predic(data)
        rtn = '["%f","%f","%f","%s","%s","%s","%f","%f","%f","%s","%s","%s"]' % ( rate1[0],rate1[1],rate1[2],code1[0],code1[1],code1[2],rate[0],rate[1],rate[2],code[0],code[1],code[2])
        return rtn

if __name__ == "__main__":
    web.wsgi.runwsgi = lambda func, addr=None: web.wsgi.runfcgi(func, addr)
    app.run()
