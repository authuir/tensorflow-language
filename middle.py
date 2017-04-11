#!/usr/local/bin/python3
#coding=utf-8

import vietnamese
import vietnamese_CNN
import os
import sys
import json

if __name__ == '__main__':
    data = json.loads(sys.argv[1])
    os.chdir('/www/yny')
    rate, code = vietnamese.predic(data)
    rate_CNN, code_CNN = vietnamese_CNN.predic(data)
    rtn = '["%f","%f","%f","%s","%s","%s","%f","%f","%f","%s","%s","%s"]' % ( rate_CNN[0],rate_CNN[1],rate_CNN[2],code_CNN[0],code_CNN[1],code_CNN[2],rate[0],rate[1],rate[2],code[0],code[1],code[2])
    print(rtn)