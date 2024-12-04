import re
import sys
from datetime import datetime, timedelta
from geopy.distance import geodesic

# site to auto-parse gps data for checking: https://swairlearn.bluecover.pt/nmea_analyser

# Constants
MIN_SPEED_THRESHOLD = 0.5  # Minimum speed in m/s for valid data
MAX_POINTS_PER_PATH = 20000  # Maximum number of points in one path
DATA_START_REGEX = r'^\$GPRMC'  # Data starts with this regex
VALID_FIX_QUALITY = {1, 2}  # Acceptable fix qualities (e.g., 1 for GPS fix, 2 for DGPS fix)


# parse all fields of GRMPC line 
def parseGRMPC(fields, parsed_line): 
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
    parsed_line["datetime"] = datetime_utc
    parsed_line["latitude"] = latitude
    parsed_line["longitude"] = longitude
    parsed_line["lat_dir"] = lat_dir
    parsed_line["long_dir"] = long_dir
    parsed_line["speed"] = speed
    parsed_line["heading"] = heading

    return parsed_line #datetime_utc, latitude, longitude, lat_dir, long_dir, speed, heading


# parse all fields of GPGGA line 
def parseGPGGA(fields, parsed_line): 
    altitude = float(fields[9]) if fields[9] else None
    fix_quality = int(fields[6]) if fields[6] else None

    parsed_line["altitude"] = altitude
    parsed_line["fix_quality"] = fix_quality
    return parsed_line


# Parse a GPS data line to extract relevant fields.
# Returns list of all parsed lines 
#   Each line captured by a dict of all its fields 
def parse_gps_line(filename):
    with open(filename, 'r') as file:
        final_data = []

        for line1 in file:
            line1.strip()
            parsed_line  = {}

            if line1.startswith('$GPRMC'):
                line2 = file.readline().strip()
                #print("line1 is :" , line1, "\nline2 is :" , line2)
                fields = line1.split(',')
                parsed_line = parseGRMPC(fields, parsed_line)

                
                #This happens if arduino eats a nl 
                if '$GPGGA' in line1:
                    parsed_line = parseGPGGA(fields, parsed_line)


                # pretty sure we only care about the altitude here
                elif line2.startswith('$GPGGA'):
                    fields = line2.split(',')
                    parsed_line = parseGPGGA(fields, parsed_line)
                    
                else: 
                    parsed_line["altitude"] = None 
                    parsed_line["fix_quality"] = None 
            
            # Capture lines that are missing GRMPC data, only have unpaired GPGGA data as NONE, 
            # since there is not enough info to only use GPGGA, this is assuming that lng will also not be recorded 
            elif not line1.startswith('lng'): 
                parsed_line["datetime"] = None
            
            print("parsedline: ", parsed_line)
            final_data.append(parsed_line)
        
        return final_data
            



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
    final_data = parse_gps_line(filename)
    '''
    with open(filename, 'r') as file:
        raw_data = file.readlines()

    parsed_data = []
    for line in raw_data:
        parsed_line = parse_gps_line(filename, raw_data, line.strip())
        print(parsed_line)
        if parsed_line and isinstance(parsed_line, tuple) and len(parsed_line) > 4:
            parsed_data.append(parsed_line)
    '''

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
