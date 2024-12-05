'''
Project 2 : Data Mining with NMEA GPS Data 
Anna Kurchenko and Lindsay Cagarli 
12/4/24

'''

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


# parse all fields from GRMPC line 
def parse_GPRMC(fields, parsed_line): 
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

    return parsed_line 


# parse all fields from GPGGA line 
def parse_GPGGA(fields, parsed_line): 
    altitude = float(fields[9]) if fields[9] else None
    fix_quality = int(fields[6]) if fields[6] else None

    parsed_line["altitude"] = altitude
    parsed_line["fix_quality"] = fix_quality
    return parsed_line

'''
 Clean and parse a GPS data line and extract relevant fields.
 Returns list of all parsed lines 
   Each line captured by a dict of all its fields 

'''
def parsed_gps_lines(filename):
    with open(filename, 'r') as file:
        final_data = []

        for line1 in file:
            line1.strip()
            parsed_line  = {}

            if line1.startswith('$GPRMC'):
                line2 = file.readline().strip()
                #print("line1 is :" , line1, "line2 is :" , line2, "\n")

                # correct case 
                if not '$GPGGA' in line1:
                    fields = line1.split(',')        
                    parsed_line = parse_GPRMC(fields, parsed_line)

                    if line2.startswith('$GPGGA'):
                        # correct case 
                        if not '$GPRMC' in line2:
                            fields = line2.split(',')        
                            parsed_line = parse_GPGGA(fields, parsed_line)

                        # This happens if arduino eats a nl 
                        elif '$GPRMC' in line2 or 'lng' in line2:
                            fields = line2.split('$')[1].split(',')        #After 1st $ before lng
                            parsed_line = parse_GPGGA(fields, parsed_line)
                            #do not use lng or next line's GRMPC if present
                    else : 
                        # Means we have no altitude, skip this data point 
                        parsed_line = {}

                # This happens if arduino eats a nl 
                elif '$GPGGA' in line1:
                    fields = line1.split('$')[1].split(',')        #After 1st $
                    parsed_line = parse_GPRMC(fields, parsed_line)

                    fields = line1.split('$')[2].split(',')        #After 2nd $ 
                    parsed_line = parse_GPGGA(fields, parsed_line)
                    #After this jumps to next final condition
            

            # pretty sure we only care about the altitude here
            elif line1.startswith('$GPGGA'):
                line2 = file.readline().strip()
                #print("line1 is :" , line1, "line2 is :" , line2, "\n")

                # correct case 
                if not '$GPRMC' in line1:
                    fields = line1.split(',')        
                    parsed_line = parse_GPGGA(fields, parsed_line)

                    if line2.startswith('$GPRMC'):
                        # correct case 
                        if not '$GPGGA' in line2:
                            fields = line2.split(',')        
                            parsed_line = parse_GPRMC(fields, parsed_line)

                        # This happens if arduino eats a nl 
                        elif ('$GPGGA' in line2) or ('lng' in line2):
                            fields = line2.split('$')[1].split(',')        #After 1st $ before lng
                            parsed_line = parse_GPRMC(fields, parsed_line)
                            #do not use lng or next line's GRMPC if present
                        
                    else : 
                        # Means we have no GRMPC data, skip this datapoint 
                        parsed_line = {}

                # This happens if arduino eats a nl 
                elif '$GPRMC' in line1:
                    fields = line1.split('$')[1].split(',')        #After 1st $
                    parsed_line = parse_GPGGA(fields, parsed_line)

                    fields = line1.split('$')[2].split(',')        #After 2nd $ 
                    parsed_line = parse_GPRMC(fields, parsed_line)
                    #After this jumps to next final condition
            

            #If no GRMPC/GPGGA line skip this data 
            #Can add in lng line parsing to minimize skipped data
            else: 
                parsed_line = {}

            # skip empty lines 
            if len(parsed_line) > 0: 
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


'''
What day did the trip occur.  Report this as YYYY/MM/DD, 
For example, 2024/09/23 for Sept 23rd. (5) 
2. What time of day did the trip start. 
Use UTC for this.  That’s fine.   
Report this as HH:MM, with HH as a 24-hour clock
'''
def trip_date_occurance(parsed_data):
    date  = parsed_data[0]["datetime"].date().strftime('%Y/%m/%d')
    print(f"GPS Trip Occured on: {date}")

    time  = parsed_data[0]["datetime"].time()
    print(f"Trip started at time : {time}")




from math import radians, sin, cos, sqrt, atan2

"""
Checks if a coordinate is within a given radius of a target location.
    
Parameters:
    coord (tuple): Tuple of (latitude, longitude) for the trip point.
    target (tuple): Tuple of (latitude, longitude) for the target location.
    radius (float): Distance in kilometers to consider 'near'. Default is 0.5 km.

"""

def is_near_location(coord, target, radius=0.5):
    # Haversine formula
    R = 6371.0  # Earth's radius in kilometers
    lat1, lon1 = radians(coord[0]), radians(coord[1])
    lat2, lon2 = radians(target[0]), radians(target[1])
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    distance = R * c

    return distance <= radius



# Determines if the trip started near a given location.
def trip_started_near_location(trip_data, target_location, radius=2):
    start_point = (trip_data[0]['latitude'], trip_data[0]['longitude'])
    return is_near_location(start_point, target_location, radius)

# Determines if the trip ended near a given location.
def trip_ended_near_location(trip_data, location_b, radius=2):
    end_point = (trip_data[-1]['latitude'], trip_data[-1]['longitude'])
    return is_near_location(end_point, location_b, radius)



'''
RIT :  *Make GPS fence for this much larger to compensate large area
82 Lomb Memorial Drive, Rochester, NY 14623
Henrietta New York United States
lat : 43.086065
long: -77.68094333

Kinsman Address: 
34 Random Knolls drive, Penfield NY 14526 - 1970 
43.138238, -77.437821

'''

# Main subroutine navigating full functionality 
def main():
    filename = sys.argv[1]
    trip_data = parsed_gps_lines(filename)

    trip_date_occurance(trip_data)

    RIT_location = (43.086065, -77.68094333)
    kinsman_res_location = (43.138238, -77.437821)

    print(trip_started_near_location(trip_data, kinsman_res_location))  # True
    print(trip_ended_near_location(trip_data, RIT_location))    # True


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
