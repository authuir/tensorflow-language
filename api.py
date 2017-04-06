#!/usr/bin/python
#coding=utf-8

import web
import json
import vietnamese
import vietnamese_CNN

urls = (
    '/api', 'Api',
    )
app = web.application(urls,globals())

class Api:
    def POST(self):  
        i = web.input().data
        data = json.loads(i)
        rate, code = vietnamese.predic(data)
        rate_CNN, code_CNN = vietnamese_CNN.predic(data)
        rtn = '["%f","%f","%f","%s","%s","%s","%f","%f","%f","%s","%s","%s"]' % ( rate_CNN[0],rate_CNN[1],rate_CNN[2],code_CNN[0],code_CNN[1],code_CNN[2],rate[0],rate[1],rate[2],code[0],code[1],code[2])
        return rtn

if __name__ == "__main__":
    web.wsgi.runwsgi = lambda func, addr=None: web.wsgi.runfcgi(func, addr)
    app.run()
