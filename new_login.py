#coding=utf-8

import requests
import sys

def login(name, passwd):
	headers={
		'User-Agent':'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.81 Safari/537.36',
		'Upgrade-Insecure-Requests':'1',
		'Connection':'keep-alive'
	}
	data = {'user':name, 'pass':passwd, 'line':"CUC-BRAS"}
	requests.post('http://ngw.bupt.edu.cn/login', data, headers=headers)
	requests.get('http://10.3.8.211/F.htm')

def logout():
	requests.get('http://ngw.bupt.edu.cn/logout')

if __name__ == '__main__':
	name = '2017111454'
	passwd = 'shanao33'
	if len(sys.argv) == 1:
		login(name, passwd)
	elif sys.argv[1] == 'in':
		login(name, passwd)
	elif sys.argv[1] == 'out':
		logout()
	else:
		print("Wrong")
