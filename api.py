#!/usr/local/bin/python3
#coding=utf-8

import web
import json
import os
import subprocess

urls = (
    '/api', 'Api',
    )
app = web.application(urls,globals())

class Api:
    def POST(self):  
        process = subprocess.Popen(['python3','/www/yny/middle.py', web.input().data],stdout=subprocess.PIPE,stderr=None)
        out_text = process.stdout.read().decode('utf-8')
        return out_text

if __name__ == "__main__":
    web.wsgi.runwsgi = lambda func, addr=None: web.wsgi.runfcgi(func, addr)
    app.run()
