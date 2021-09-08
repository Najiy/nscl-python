import sys, csv, os, shutil, time
from pathlib import Path
from datetime import datetime
from collections import deque
import os, time

sensor_directory = r"G:\human_activity_raw_sensor_data\compiled"
newest = datetime.fromisoformat("2020-01-01 00:00:00.000")

print(
    "\n   Data Collator: Multi-sensor dataset of human activities in a smart home environment. \n     (https://data.mendeley.com/datasets/t9n68ykfk3/1)\n"
)

mergeddir = sensor_directory + r"\mergedset"


def renewdir(dirpath):
    if os.path.exists(dirpath):
        shutil.rmtree(dirpath)
    os.mkdir(dirpath)


renewdir(mergeddir)

t = time.time()

for month in range(1, 13):
    mergedset = {}
    counter = 0
    mfile = r"%s\%s.csv" % (mergeddir, month)

    for sensor in os.listdir(sensor_directory):
        f = r"%s\%s\%s.csv" % (sensor_directory, sensor, month)

        if os.path.exists(f):

            # try:
            with open(f, "r") as fhandler:
                print("opened", f)

                reader = csv.DictReader(fhandler)
                for line in reader:
                    tstamp = str(
                        datetime.fromisoformat(line["timestamp"]).replace(microsecond=0)
                    )

                    # if tstamp not in mergedset.keys():
                    #     mergedset[tstamp] = []

                    if float(line["delta"]) != 0.0:
                        if tstamp not in mergedset.keys():
                            mergedset[tstamp] = []
                        mergedset[tstamp].append(
                            (sensor, float(line["delta"]), float(line["value"]))
                        )
                        counter += 1

        # except FileNotFoundError as e:
        #     print(e)
        #     continue

    with open(mfile, "w") as fhandler:
        fieldnames = ["timestamp", "sensors", "deltas", "values"]
        writer = csv.DictWriter(fhandler, fieldnames)
        for tstamp in mergedset:
            sensors = "+".join([str(x[0]) for x in mergedset[tstamp]])
            deltas = "+".join([str(x[1]) for x in mergedset[tstamp]])
            values = "+".join([str(x[2]) for x in mergedset[tstamp]])
            entry = {
                "timestamp": tstamp,
                "sensors": sensors,
                "deltas": deltas,
                "values": values,
            }
            # print(entry)
            writer.writerow(entry)

    print(
        "     month=%s  count=%s  t_elapsed=%s"
        % (month, counter, (time.time() - t) / 60)
    )

print()
print("     done %s " % ((time.time() - t) / 60))
