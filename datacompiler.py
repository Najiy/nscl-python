import sys, csv, os, shutil, time
from pathlib import Path
from datetime import datetime


print(
    "\n   Data Compiler: Multi-sensor dataset of human activities in a smart home environment. \n     (https://data.mendeley.com/datasets/t9n68ykfk3/1)\n"
)

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
        sens_name = r"\%s.csv" % (month)
        sens_path = r"%s\%s\%s" % (sensor_compile_directory, sensor[3], sens_name)
        sens_file = open(sens_path, "a+")
        sens_file.write("%s,%s,%s\n" % ("timestamp", "value", "delta"))

        oldest = datetime.today()
        newest = datetime.fromisoformat("2020-01-01 00:00:00.000")
        tstart = time.time()
        minimum = 0
        maximum = 0
        prev = 0

        for line in reader:
            if line["sensor_id"] == sensor[0]:

                records += 1

                timestamp = datetime.fromisoformat(line["timestamp"])
                value = float(line["value"])

                if timestamp.month != month:
                    month = timestamp.month
                    sens_file.close()
                    sens_name = r"\%s.csv" % (month)
                    sens_path = r"%s\%s\%s" % (
                        sensor_compile_directory,
                        sensor[3],
                        sens_name,
                    )
                    sens_file = open(sens_path, "a+")
                    sens_file.write("%s,%s,%s\n" % ("timestamp", "value", "delta"))

                sens_file.write(
                    "%s,%s,%s\n" % (line["timestamp"], line["value"], str(value - prev))
                )

                prev = float(line["value"])

                if records == 1:
                    minimum = value
                    maximum = value
                elif value < minimum:
                    minimum = value
                elif value > maximum:
                    maximum = value

                if timestamp < oldest:
                    oldest = timestamp
                elif timestamp > newest:
                    newest = timestamp

                if records % 100000 == 0:
                    print(
                        r"%.2f" % ((time.time() - t) / 60),
                        sensor,
                        records,
                        oldest,
                        newest,
                        minimum,
                        maximum,
                    )
                # info(sensor, line)

        elapsed = (time.time() - tstart) / 60

        meta_file = open(sensor_compile_directory + r"\metadata.csv", "a+")
        meta_file.write(
            "%s,%s,%s,%s,%s,%s,%s\n"
            % (sensor[3], records, elapsed, oldest, newest, minimum, maximum)
        )

        sens_file.close()
        meta_file.close()
        # with open("%s_%s" % (sensor[0], sensor[3].replace("/", "_")), 'rw') as sfile:

    # elif sensor[2] == 'FLOAT':


with open(sensor_list, "r", newline="") as fhand:
    reader = csv.DictReader(fhand)
    t = time.time()

    carryon = input("   carry on? y/n: ")

    if carryon == "y":
        try:
            shutil.rmtree(sensor_compile_directory)
        except OSError as e:
            print("OSError as e: %s : %s" % (sensor_compile_directory, e.strerror))

        os.mkdir(sensor_compile_directory)

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
        print("aborted")
