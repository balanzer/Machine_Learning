
import sys


def main():
    try:
        x = int('123')/0
        print(f"x value is {x}")
    except ValueError:
        print('invalid number')
    except:
        print(f'unknown error  - details {sys.exc_info()}')
        raise TypeError('do valid operations ' + sys.exc_info())
    else:
        print(f'all executed well x is {x}, this is final block')


main()
