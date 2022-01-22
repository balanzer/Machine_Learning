import sys
import os
import datetime
import greetings_module


def printVersion():
    v = sys.version
    vi = sys.version_info
    print(f' version is {v}, version info is {vi} ')
    print(f' os name is {os.name} ')


def printDate():
    now = datetime.datetime.now()
    print(f'now is {now}')


if __name__ == '__main__':
    print('testing changes')
    # printVersion()
    # printDate()
    welcome = greetings_module.greetings('MMMM')
    welcome.sayHello()
