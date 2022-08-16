from dataclasses import replace
import random
import numpy as np

interval = 180
padding = 4
int_streams = 10
float_streams = 0

choices = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
int_array = [f"stream{x}" for x in range(int_streams)]
float_array = [f"fstream{x}" for x in range(float_streams)]


randoms_int = [
    [round(random.choice(choices)) for _ in range(0, interval)] for __ in int_array
]

randoms_float = [
    [
        int(random.uniform(0, 1024)) if random.choice(choices) == 1 else 0
        for x in range(0, interval)
    ]
    for __ in float_array
]


# print(int_array + float_array)
# print(randoms_int + randoms_float)

headers = int_array + float_array
joint_array = randoms_int + randoms_float

lines = []

lines.append("unix_time,"+",".join(headers) + ",\n")
for i in range(interval):
    str_temp = f"{i},"
    for j in joint_array:
        if j[i] == 0:
            str_temp += f","
        else:
            str_temp += f"{j[i]},"
    lines.append(str_temp + "\n")


f = open("./dataset/dataset.csv", "w")
f.writelines(lines)


######################################################################


meta_headers = "sensor,records,elapsed,unix_oldest,unix_newest,oldest,newest,minimum,maximum,min2,max2,min3,max3,\n"

meta_line = "name,,,,,,,,,min,max,,,"
meta_content = [meta_headers]

for h in headers:
    if "fstream" in h:
        meta_content.append(
            meta_line.replace("name", h).replace("min", "0").replace("max", "1024")
            + "\n"
        )
    elif "stream" in h:
        meta_content.append(
            meta_line.replace("name", h).replace("min", "0").replace("max", "1") + "\n"
        )

print(meta_content)


f = open("./dataset/meta.csv", "w")
f.writelines(meta_content)
