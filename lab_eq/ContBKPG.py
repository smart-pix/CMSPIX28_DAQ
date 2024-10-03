#!/bin/python3
import os
import time
import io
d = os.open('/dev/usbtmc0', os.O_RDWR)
input = [
"*IDN?",
"C1:BSWV WVTP,PULSE",
"C1:BSWV FRQ,1000HZ",
"C1:BSWV PERI,0.001S",
"C1:BSWV HLEV,0.2V",
"C1:BSWV LLEV,0V",
"C1:BSWV DUTY,20",
"C1:BSWV RISE,6e-09S",
"C1:BSWV FALL,6e-09S",
"C1:BSWV DLY,0",
"C1:BSWV?",
"C1:BTWV STATE,ON",
"C1:BTWV TRSR,EXT",
"C1:BTWV TIME,1",
"C1:BTWV DLAY,6.75e-07S",
"C1:BTWV EDGE,FALL",
"C1:BTWV CARR,WVTP,PULSE",
"C1:BTWV FRQ,1000HZ",
"C1:BTWV PERI,0.001S",
"C1:BTWV HLEV,0.2V",
"C1:BTWV LLEV,0V",
"C1:BTWV DUTY,20",
"C1:BTWV RISE,6e-09S",
"C1:BTWV FALL,6e-09S",
"C1:BTWV DLY,0",
"C1:BTWV?",
"C1:OUTP ON",
"C1:OUTP LOAD,HZ"
]
nlist=len(input)
for i in range(nlist): 
    os.write(d,input[i].encode())
    out = b' '
    # let's wait one second before reading output (let's give device time to answer)
    print(input[i])
    if(input[i][-1]=="?"):   #If the last character of the request is a question 
        out=os.read(d,1024)  #Print out the response
        print(out.decode())

