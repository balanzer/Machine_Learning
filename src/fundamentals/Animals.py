class Animal:
    def __init__(self, type, name, sound):
        self._type = type
        self._name = name
        self._sound = sound

    def type(self, type=None):
        if type:
            self._type = type
        return self._type

    def name(self, name=None):
        if name:
            self._name = name
        return self._name

    def sound(self, sound=None):
        if sound:
            self._sound = sound
        return self._sound


def main():
    tiger = Animal('cat', 'Tiger', 'Roar')
    wolf = Animal('Dog', 'Wolf', 'Howl')
    printAnimals(wolf)
    printAnimals(tiger)
    printAnimals(None)


def printAnimals(o):
    if not isinstance(o, Animal):
        raise TypeError('input is not Animal Type')
    print(f"{o.name()} is a {o.type()} and sounds is {o.sound()}")


main()
