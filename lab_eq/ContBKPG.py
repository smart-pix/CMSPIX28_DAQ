#!/bin/python3
import os
import time
import io
d = os.open('/dev/usbtmc0', os.O_RDWR)

print('Enter your commands below.\r\nInsert "exit" to leave the application.')

while 1 :
    # get keyboard input
    #input = raw_input(">> ")
        # Python 3 users
    instr = input(">> ")
    if instr == 'exit':
        os.close(d)
        exit()
    else:
        # send the string to the device
        input2 = instr + '\n'
        os.write(d,input2.encode())
        out = b' '
        # let's wait one second before reading output (let's give device time to answer)
        time.sleep(1)
        out=os.read(d,1024)
        print(">>" + out.decode())

