import random


class greetings:
    def __init__(self, name=None):
        self.__name = name

    def sayHello(self):
        if self.__name:
            print(
                f"Hello {self.__name}, Greetings!. Random Number for you is {random.randint(1,2000)}")
        else:
            print(
                f"Hello stranger, Greetings!. No random number for you")


if __name__ == '__main__':
    stranger = greetings()
    me = greetings('Murali')

    print(stranger.sayHello())
    print(me.sayHello())
