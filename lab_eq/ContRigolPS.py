#!/usr/bin/python3
import io
import time
import serial

# configure the serial connections (the parameters differs on the device you are connecting to)
ser = serial.Serial(
    port='/dev/ttyS0',timeout=1
)

ser.close()
ser.open()

print('Enter your commands below.\r\nInsert "exit" to leave the application.')

while 1 :
    # get keyboard input
    #input = raw_input(">> ")
        # Python 3 users
    instr = input(">> ")
    if instr == 'exit':
        ser.close()
        exit()
    else:
        # send the character to the device
        input2 = instr + '\n'
        ser.write(input2.encode())
        out = b' '
        # let's wait one second before reading output (let's give device time to answer)
        time.sleep(1)
        while ser.inWaiting() > 0:
            out += ser.read(1)
            
        if out != '':
            print(">>" + out.decode())
