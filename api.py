#!/usr/bin/python
#coding=utf-8

import web
import subprocess

urls = (
    '/api', 'Api',
    )
app = web.application(urls,globals())

class Api:
    def POST(self):  
        i = web.input().data
        process = subprocess.Popen(['python','./vietnamese.py',i],stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        out_text = process.stdout.read().decode('utf-8')
        rtn = ""
        rtn = rtn+out_text.split('\r')[0]
        process = subprocess.Popen(['python','./vietnamese_CNN.py',i],stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        out_text = process.stdout.read().decode('utf-8')
        rtn += out_text.split('\r')[0]
        return rtn

if __name__ == "__main__":
    web.wsgi.runwsgi = lambda func, addr=None: web.wsgi.runfcgi(func, addr)
    app.run()
