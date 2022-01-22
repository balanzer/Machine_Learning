def openAndPrint():

    fName = 'lines.txt'
    try:
        f = open(fName, 'r')
        print(f'Opening {fName}')

        for line in f:
            print(line.rstrip())

    except Exception as err:
        print(f'reading file {fName} ends in error : {err}')


openAndPrint()
