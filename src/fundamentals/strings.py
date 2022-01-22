
st = 'Hello World'

print(f'Upper : {st.upper()}')
print(f'swapcase : {st.swapcase()}')
print(f'casefold : {st.casefold()}')


class MyStr(str):
    def __str__(self):
        return self[::-1]


st = MyStr('Hello World, Greetings')

print(st)

# Common methods

st = 'Hello World im learning python'

print(f'Upper : {st.upper()}')
print(f'swapcase : {st.swapcase()}')
print(f'capitalize : {st.capitalize()}')
print(f'casefold : {st.casefold()}')

# Formating

x = "2*2 is {}".format(2*2)
print(x)


y = "two numbers with left are right justfications are {0:>05} {1:>05} ends here".format(
    21, 33)
print(y)

#Split and Joins

st = 'Hello World im learning python'
lst = st.split(' ')
print(lst)
st2 = ':'.join(lst)
print(st2)


# prints ascii char
for s in st:
    print(f'{s} ascii is {ord(s)}, next char of s is {chr(ord(s)+1)}')
