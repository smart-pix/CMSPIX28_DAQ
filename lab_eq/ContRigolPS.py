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
input = [
"*IDN?",
"*IDN?",
"*IDN?"
#"MEAS:ALL?"
#":VOLT 0.9",
#":CURR 0.03",
#":CURR:PROT 0.04",
#":OUTP:STAT CH1,ON",
#"MEAS:ALL?",
]
nlist=len(input)
for i in range(nlist):
    ser.write(input[i].encode())
    out = b' '
    time.sleep(1)
    print(input[i])
    out += ser.read(1)
    print(out)
    print(out.decode())
"""     out += ser.read(1)   
    if(input[i][-1]=="?"):
        print(">>" + out.decode())
ser.close() """
