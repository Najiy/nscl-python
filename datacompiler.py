from contextlib import contextmanager
from re import M
import sys, csv, os, shutil, time
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import pprint

# newest = datetime.fromisoformat("2020-01-01 00:00:00.830")
# unixtime = int(time.mktime(newest.timetuple()))
# print(unixtime)
# newest.replace(microsecond=0)
# unixtime = int(time.mktime(newest.timetuple()))
# print(unixtime)

print(
    "\n   Data Compiler: Multi-sensor dataset of human activities in a smart home environment. \n     (https://data.mendeley.com/datasets/t9n68ykfk3/1)\n"
)

pp = pprint.PrettyPrinter(indent=4)

sensor_directory = r"G:\human_activity_raw_sensor_data"
sensor_alt_directory = r"C:\Users\najiy\Desktop\human_activity_raw_sensor_data"

select = sensor_directory

sensor_compile_directory = select + r"\compiled"
sensor_list = select + r"\sensor.csv"
sensor_sample_float = select + r"\sensor_sample_float.csv"
sensor_sample_int = select + r"\sensor_sample_int.csv"


def info(sensor, line):
    if line["value"] != "0":
        print(sensor[3], line["value"], type(line))


def compile_sensor(sensor, t):
    records = 0
    sample_file = ""

    if sensor[2] == "INT":
        sample_file = sensor_sample_int
    elif sensor[2] == "FLOAT":
        sample_file = sensor_sample_float

    with open(sample_file, "r", newline="") as fhand:
        reader = csv.DictReader(fhand)
        os.mkdir(r"%s\%s" % (sensor_compile_directory, sensor[3]))

        month = 2
        # sens_name = r"\%s.csv" % (sensor[3])
        # sens_path = sensor_compile_directory + sens_name
        sens_name = r"\%s.csv" % ("data")  # (month)
        sens_path = r"%s\%s\%s" % (sensor_compile_directory, sensor[3], sens_name)
        newfile = True

        if os.path.isfile(sens_path):
            newfile = False

        sens_file = open(sens_path, "a+")

        if newfile:
            sens_file.write(
                "%s,%s,%s,%s\r" % ("timestamp", "unix_timestamp", "value", "delta")
            )

        oldest = datetime.today()
        newest = datetime.fromisoformat("2020-01-01 00:00:00.000")
        unixoldest = int(time.mktime(oldest.timetuple()))
        unixnewest = int(time.mktime(newest.timetuple()))
        tstart = time.time()
        unixtime = 0
        minimum = 0
        maximum = 0
        max2 = 0
        max3 = 0
        min2 = 0
        min3 = 0
        prev = 0

        for line in reader:
            if line["sensor_id"] == sensor[0]:

                records += 1

                timestamp = datetime.fromisoformat(line["timestamp"])
                value = float(line["value"])

                if timestamp.month != month:
                    month = timestamp.month
                    sens_file.close()
                    sens_name = r"\%s.csv" % ("data")  # (month)
                    sens_path = r"%s\%s\%s" % (
                        sensor_compile_directory,
                        sensor[3],
                        sens_name,
                    )
                    newfile = True
                    if os.path.isfile(sens_path):
                        newfile = False
                    sens_file = open(sens_path, "a+")
                    if newfile:
                        sens_file.write(
                            "%s,%s,%s,%s\r"
                            % ("timestamp", "unix_timestamp", "value", "delta")
                        )

                if value - prev != 0:
                    u_time = datetime.fromisoformat(line["timestamp"])
                    unixtime = int(time.mktime(u_time.timetuple()))
                    sens_file.write(
                        "%s,%s,%s,%s\r"
                        % (
                            line["timestamp"],
                            unixtime,
                            line["value"],
                            str(value - prev),
                        )
                    )

                prev = float(line["value"])

                if records == 1:
                    minimum = value
                    maximum = value
                    min2 = value
                    max2 = value
                    min3 = value
                    max3 = value
                elif value < minimum:
                    min3 = min2
                    min2 = minimum
                    minimum = value
                elif value > maximum:
                    max3 = max2
                    max2 = maximum
                    maximum = value

                if unixtime < unixoldest and unixtime != 0:
                    unixoldest = unixtime
                elif unixtime > unixnewest:
                    unixnewest = unixtime

                if timestamp < oldest:
                    oldest = timestamp
                elif timestamp > newest:
                    newest = timestamp

                if records % 100000 == 0:
                    print(
                        r"%.2f" % ((time.time() - t) / 60),
                        sensor,
                        records,
                        unixoldest,
                        unixnewest,
                        oldest,
                        newest,
                        minimum,
                        min2,
                        min3,
                        maximum,
                        max2,
                        max3,
                    )
                # info(sensor, line)

        elapsed = (time.time() - tstart) / 60

        meta_file = open(sensor_compile_directory + r"\metadata.csv", "a")

        meta_file.write(
            "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\r"
            % (
                sensor[3],
                records,
                elapsed,
                unixoldest,
                unixnewest,

                oldest,
                newest,
                minimum,
                min2,
                min3,

                maximum,
                max2,
                max3,
            )
        )

        sens_file.close()
        meta_file.close()
        # with open("%s_%s" % (sensor[0], sensor[3].replace("/", "_")), 'rw') as sfile:

    # elif sensor[2] == 'FLOAT':


with open(sensor_list, "r", newline="") as fhand:
    reader = csv.DictReader(fhand)
    t = time.time()

    data_compile = input("   compile sensor data? y/n: ")

    if data_compile == "y":
        try:
            shutil.rmtree(sensor_compile_directory)
        except OSError as e:
            print("OSError as e: %s : %s" % (sensor_compile_directory, e.strerror))

        os.mkdir(sensor_compile_directory)

        meta_file = open(sensor_compile_directory + r"\metadata.csv", "w")
        meta_file.write(
            "sensor, records, elapsed, unix_oldest, unix_newest, oldest, newest, minimum, maximum, min2, max2, min3, max3\r"
        )
        meta_file.close()

        print()

        for row in reader:
            # sensor_id, node_id, type, name
            sensor = (
                row["sensor_id"],
                row["node_id"],
                row["type"],
                row["name"].replace(" ", "").replace("/", "_"),
            )
            compile_sensor(sensor, t)

        print()
        print("     done %s " % ((time.time() - t) / 60))

    else:
        print("     ...compile skipped")

    ########################################################

    data_merge = input("   merge sensor data? y/n: ")

    if data_merge == "y":

        meta_data = pd.read_csv(
            sensor_compile_directory + r"\metadata.csv",
            # sep="\r",
            delimiter=",",
            dtype={" unix_oldest": np.float64, " unix_newest": np.int32},
        )

        unix_time_min = int(meta_data[" unix_oldest"].min())
        unix_time_max = int(meta_data[" unix_newest"].max())

        print("min", unix_time_min)
        print("max", unix_time_max)

        # print("min2", )
        # print("max2", )
        # print("min3",)
        # print("max3", )

        sensordirs = [
            d
            for d in os.listdir(sensor_compile_directory)
            if os.path.isdir(sensor_compile_directory + "/" + d)
        ]

        fields = []
        contents = {}

        for s in sensordirs:
            fields.append(s)
            file_df = pd.read_csv(sensor_compile_directory + "/" + s + "/data.csv")

            print("reading " + s)
            for idx, tstamp in enumerate(file_df["unix_timestamp"]):
                if tstamp not in contents:
                    contents[tstamp] = {}
                cvalue = contents[tstamp]
                cvalue[s] = file_df["value"][idx]

        # print("merging")

        mergedfile = open(sensor_compile_directory + "/merged.csv", "w")

        mergedfile.write("unix_time,")
        for s in sensordirs:
            mergedfile.write(f"{s},")
        mergedfile.write("\r")

        for i in range(unix_time_min, unix_time_max):
            if i % 500 == 0:
                os.system("cls")
                print(
                    f"merging {(i-unix_time_min) / (unix_time_max-unix_time_min) * 100: .1f}%"
                )

            if i not in contents:
                continue

            line = f"{i},"

            for s in sensordirs:
                if s in contents[i]:
                    line += "%s," % contents[i][s]
                else:
                    line += ","

            line += "\r"
            mergedfile.write(line)

        mergedfile.close()

        # pp.pprint()
        input("done")

    else:
        print("merge skipped")
