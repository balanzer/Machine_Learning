
x = '47'
y = -1
z = -1
try:
    y = int(x)
except ValueError as err:
    print(f'invalid input to converted to integer')

try:
    z = float(x)
except ValueError as err:
    print(f'invalid input to converted to float')


print(f'values x is {x}, y is {y}, z is {z}')
print(f'type of x is {type(x)}, y is {type(y)}, z is {type(z)}')


divModVal = divmod(y, 3)

print(f'divmod for {y} and 3 is {divModVal}')
