import re
import sys
from datetime import datetime, timedelta
from geopy.distance import geodesic

# Constants
MIN_SPEED_THRESHOLD = 0.5  # Minimum speed in m/s for valid data
MAX_POINTS_PER_PATH = 20000  # Maximum number of points in one path
DATA_START_REGEX = r'^\$GPRMC'  # Data starts with this regex
VALID_FIX_QUALITY = {1, 2}  # Acceptable fix qualities (e.g., 1 for GPS fix, 2 for DGPS fix)


def parse_gps_line(line):
    """
    Parse a GPS data line to extract relevant fields.
    """
    try:
        if line.startswith('$GPRMC'):
            fields = line.split(',')
            time_utc = fields[1]
            status = fields[2]
            latitude = float(fields[3]) if fields[3] else None
            lat_dir = fields[4]
            longitude = float(fields[5]) if fields[5] else None
            long_dir = fields[6]
            speed = float(fields[7]) * 0.514444  # Convert knots to m/s
            heading = float(fields[8]) if fields[8] else None
            date = fields[9]

            if latitude and lat_dir == 'S':
                latitude = -latitude
            if longitude and long_dir == 'W':
                longitude = -longitude

            datetime_utc = datetime.strptime(date + time_utc[:6], '%d%m%y%H%M%S')
            return datetime_utc, latitude, longitude, speed, heading, status == 'A'
        elif line.startswith('$GPGGA'):
            fields = line.split(',')
            altitude = float(fields[9]) if fields[9] else None
            fix_quality = int(fields[6]) if fields[6] else None
            return altitude, fix_quality
        elif line.startswith("lng="):
            match = re.match(r'lng=([\-\d\.]+), lat=([\-\d\.]+),.*speed=([\d\.]+)', line)
            if match:
                longitude, latitude, speed = map(float, match.groups())
                return latitude, longitude, speed
        return None
    except ValueError:
        return None


#gotten from website in writeup 
def parseGPS(data):
#    print "raw:", data #prints raw data
    if data[0:6] == "$GPRMC":
        sdata = data.split(",")
        if sdata[2] == 'V':
            print("no satellite data available")
            return
        print("---Parsing GPRMC---")
        time = sdata[1][0:2] + ":" + sdata[1][2:4] + ":" + sdata[1][4:6]
        lat = decode(sdata[3]) #latitude
        dirLat = sdata[4]      #latitude direction N/S
        lon = decode(sdata[5]) #longitute
        dirLon = sdata[6]      #longitude direction E/W
        speed = sdata[7]       #Speed in knots
        trCourse = sdata[8]    #True course
        date = sdata[9][0:2] + "/" + sdata[9][2:4] + "/" + sdata[9][4:6]#date
        print("time : %s, latitude : %s(%s), longitude : %s(%s), speed : %s, True Course : %s, Date : %s" %  (time,lat,dirLat,lon,dirLon,speed,trCourse,date))
def decode(coord):
    #Converts DDDMM.MMMMM > DD deg MM.MMMMM min
    x = coord.split(".")
    head = x[0]
    tail = x[1]
    deg = head[0:-2]
    min = head[-2:]
    return deg + " deg " + min + "." + tail + " min"

def filter_data(data):
    """
    Clean and filter the GPS data to remove redundant and erroneous points.
    """
    filtered_data = []
    last_point = None

    for point in data:
        if last_point:
            distance = geodesic((last_point[1], last_point[2]), (point[1], point[2])).meters
            if point[3] < MIN_SPEED_THRESHOLD or distance < 1:  # Ignore stationary points
                continue
        filtered_data.append(point)
        last_point = point

    return filtered_data


def compute_drive_duration(data):
    """
    Compute the duration of the drive from valid start to stop.
    """
    if len(data) < 2:
        return None

    start_point = data[0]
    end_point = data[-1]

    # Ensure the vehicle is moving at start and end
    if start_point[3] < MIN_SPEED_THRESHOLD or end_point[3] < MIN_SPEED_THRESHOLD:
        return None

    duration = end_point[0] - start_point[0]
    return duration


def split_paths(data):
    """
    Split the data into multiple paths if exceeding maximum points limit.
    """
    paths = []
    for i in range(0, len(data), MAX_POINTS_PER_PATH):
        paths.append(data[i:i + MAX_POINTS_PER_PATH])
    return paths


def main():
    filename = sys.argv[1]

    with open(filename, 'r') as file:
        raw_data = file.readlines()

    parsed_data = []
    for line in raw_data:
        parsed = parseGPS(line.strip())
        if parsed and isinstance(parsed, tuple) and len(parsed) > 4:
            parsed_data.append(parsed)
    
    '''
    filtered_data = filter_data(parsed_data)
    duration = compute_drive_duration(filtered_data)

    if duration:
        print(f"Drive Duration: {duration}")
    else:
        print("Invalid data: Could not compute drive duration.")

    paths = split_paths(filtered_data)
    for i, path in enumerate(paths, 1):
        print(f"Path {i}: {len(path)} points")
    '''

#if __name__ == "__main__":
#    main()

main()
