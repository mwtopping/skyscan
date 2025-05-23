import psycopg2
import pandas as pd
from datetime import datetime

from skyfield.api import load, wgs84, EarthSatellite
from skyfield.iokit import parse_tle_file


# Establish connection
conn = psycopg2.connect(
    host="localhost",
    database="skyscan",
    user="michael",
    password=""
)

cursor = conn.cursor()

#CREATE TABLE satellites ( #        id int PRIMARY KEY,
#        name TEXT NOT NULL,
#        created_at TIMESTAMP NOT NULL,
#        updated_at TIMESTAMP NOT NULL,
#        epoch TIMESTAMP NOT NULL,
#        line1 TEXT,
#        line2 TEXT
#);

# Insert a single row
insert_query = """
INSERT INTO satellites (id, name, created_at, updated_at, epoch, line1, line2) 
VALUES (%s, %s, %s, %s, %s, %s, %s)
"""


def insert_satellite(cursor, sat, line1, line2):
    # Create a datetime object
#    timestamp_value = datetime(2025, 5, 23, 14, 30, 0)  # Year, month, day, hour, minute, second

    # Or get current time
    current_time = datetime.now()
    id = sat.model.satnum
    name = sat.name

    epoch = sat.epoch.utc_datetime()
    print(id, name)
#    line1 = sat.model.line1
#    line2 = sat.model.line2

    cursor.execute(insert_query, (id, name, current_time, current_time, epoch, line1, line2))

    # Commit the transaction
    conn.commit()

ts = load.timescale()
#with load.open('./data/utc2025apr17_u.dat') as f:
#    sats = list(parse_tle_file(f, ts))

with open('./data/utc2025apr17_u.dat', 'r') as f:
    while True:
        line = f.readline().strip()
        line2 = f.readline().strip()
        line3 = f.readline().strip()

        name = line[2:]
        sat = EarthSatellite(line2, line3, name)

        if line == "":
            break

#for sat in sats:
        insert_satellite(cursor, sat, line2, line3)
# Close connections
cursor.close()
conn.close()
