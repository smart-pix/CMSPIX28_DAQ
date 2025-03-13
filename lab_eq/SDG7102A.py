import time
import os
import numpy as np
def SDG7102A_SWEEP_ORIGIN(HLEV=0.3):

    d = os.open('/dev/usbtmc0', os.O_RDWR)
    input = [
    f"C1:BSWV HLEV,{HLEV}V",
    "C1:BSWV LLEV,0V",
    ]
    nlist=len(input)
    for i in range(nlist): 
        os.write(d,input[i].encode())
        out = b' '
        # let's wait one second before reading output (let's give device time to answer)
        #print(input[i])
        if(input[i][-1]=="?"):   #If the last character of the request is a question 
            time.sleep(1)
            out=os.read(d,1024)  #Print out the response
            print(out.decode())
    os.close(d)

def SDG7102A_SWEEP(HLEV=0.2):
    input_commands = [
        f"C1:BSWV HLEV,{HLEV}V",  # Set high-level voltage
        # "C1:BSWV LLEV,0V",  # Set low-level voltage
    ]
    
    try:
        # Open the device file in read/write binary mode using 'with'
        with open('/dev/usbtmc0', 'r+b') as d:
            for cmd in input_commands:
                d.write(cmd.encode())  # Send command to device
                # Only wait and read response if command ends with "?"
                if cmd.endswith("?"):
                    time.sleep(1)  # Give the device time to respond
                    out = d.read(1024)  # Read the response
                    print(out.decode())  # Print the decoded output
                
                # If the command is not a query, we just continue
                else:
                    out = b''  # No output for non-query commands

    except Exception as e:
        print(f"Error communicating with the device: {e}")

v_min = 0.001
v_max = 0.4
v_step = 0.001
n_step = int((v_max - v_min)/v_step)+1
stepList = np.linspace(v_min, v_max, n_step)

# for v in stepList:
#     print("first code sweep test")
#     print(v)
#     SDG7102A_SWEEP_ORIGIN()

for v in stepList:
    print("second code sweep test")
    print(v)
    SDG7102A_SWEEP(0.4)


