class Duck:

    movement = 'Walk'
    sound = 'Quack Quack'

    def quack(self):
        print(f"sound is {self.sound}")

    def move(self):
        print(f"move is {self.movement}")


def main():
    donald = Duck()
    donald.move()
    donald.quack()


if __name__ == '__main__':
    main()
