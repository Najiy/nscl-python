from dataclasses import replace
import math
import random
import numpy as np
import os



interval = 600
int_streams = 5
float_streams = 5
f_tagname= "S5F5"

total_streams = int_streams + float_streams



xs = [int(x%(int_streams+float_streams)) for x in range(interval)]

for i in xs:
    print(i)

input()



# choices = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
int_array = [f"S{x}" for x in range(int_streams)]
float_array = [f"F{x}" for x in range(float_streams)]
float_masks = [int(random.uniform(float_streams+int_streams, 256)) for x in range(total_streams)]


headers = int_array + float_array

lines = []
lines.append("unix_time,"+",".join(headers) + ",\n")

for i,v in enumerate(xs):
    line = ["" for x in range(total_streams)]
    line.insert(0, f"{i}")
    if v > len(int_array):
        line[v+1] = f"{float_masks[v]}"
    else:
        line[v+1] = "1"
    lines.append(",".join(line)+",\n")

f = open(f"./dataset/dataset_saw_{f_tagname}.csv", "w")
f.writelines(lines)


######################################################################


meta_headers = "sensor,records,elapsed,unix_oldest,unix_newest,oldest,newest,minimum,maximum,min2,max2,min3,max3,\n"

meta_line = "name,,,,,,,,,min,max,,,"
meta_content = [meta_headers]

for h in headers:
    if "F" in h:
        meta_content.append(
            meta_line.replace("name", h).replace("min", "0").replace("max", "1024")
            + "\n"
        )
    elif "S" in h:
        meta_content.append(
            meta_line.replace("name", h).replace("min", "0").replace("max", "1") + "\n"
        )

print(meta_content)


f = open(f"./dataset/meta_saw_{f_tagname}.csv", "w")
f.writelines(meta_content)
