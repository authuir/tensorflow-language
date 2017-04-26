#!/usr/local/bin/python3
#coding=utf-8

from http.server import BaseHTTPRequestHandler
from http.server import SimpleHTTPRequestHandler
from http.server import HTTPServer
from socketserver import ThreadingMixIn
import urllib.parse
import json
import subprocess
import os

global path
hostIP = '127.0.0.1'
portNum = 80
path = os.getcwd()
os.chdir(os.getcwd()+'/www')
class mySoapServer( SimpleHTTPRequestHandler ):
    def do_POST( self ):
        try:
            global path
            os.chdir(path)
            content_len = int(self.headers['content-length'])
            post_body = self.rfile.read(content_len)
            self.send_response( 200, message = None )
            self.send_header( 'Content-type', 'text/html' )
            self.end_headers()
            if post_body[0:5] == b'data=':
                j_data = urllib.parse.unquote(str(post_body[5:]))
                j_data = j_data[2:len(j_data)-1]

                # Method 1
                import Softmax
                import CNN
                data = json.loads(j_data)
                rate, code = Softmax.predic(data)
                rate_CNN, code_CNN = CNN.predic(data)
                out_text = '["%f","%f","%f","%s","%s","%s","%f","%f","%f","%s","%s","%s"]' % ( rate_CNN[0],rate_CNN[1],rate_CNN[2],code_CNN[0],code_CNN[1],code_CNN[2],rate[0],rate[1],rate[2],code[0],code[1],code[2])
                self.wfile.write( out_text.encode( encoding = 'utf_8', errors = 'strict' ) )
            else:
                res = 'error'
                self.wfile.write( res.encode( encoding = 'utf_8', errors = 'strict' ) )
        except IOError:
            self.send_error( 404, message = None )
 
class ThreadingHttpServer( ThreadingMixIn, HTTPServer ):
    pass
     
myServer = ThreadingHttpServer( ( hostIP, portNum ), mySoapServer )
myServer.serve_forever()
myServer.server_close()