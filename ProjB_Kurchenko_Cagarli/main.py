'''
Project 2 : Data Mining with NMEA GPS Data 
Anna Kurchenko and Lindsay Cagarli 
12/4/24

'''

import sys
from fastkml import kml
from fastkml.styles import Style, LineStyle
from shapely.geometry import LineString, Point
from datetime import datetime, timedelta
from geopy.distance import geodesic
from math import radians, sin, cos, sqrt, atan2, atan, degrees, sqrt


# Site to auto-parse gps data for checking: https://swairlearn.bluecover.pt/nmea_analyser

# Constants
MIN_SPEED_THRESHOLD = 0.5                   # Minimum speed in m/s for valid data
MAX_POINTS_PER_PATH = 20000                 # Maximum number of points in one path
MIN_STOP_DURATION = timedelta(seconds=30)   # Minimum time stationary to be considered a stop
MAX_STOP_DURATION = timedelta(minutes=5)    # Maximum time stationary to be considered a stop
JUMP_THRESHOLD = 35                         # Threshold for a "sudden jump" (in meters)
STOP_SPEED = 5.0 * 0.44704                  # Maximum speed to be considered a stop (5 mph converted to m/s)
BRAKE_THRESHOLD = 0.5                       # Threshold to determine a harder brake
DATA_START_REGEX = r'^\$GPRMC'              # Data starts with this regex
VALID_FIX_QUALITY = {1, 2}                  # Acceptable fix qualities (e.g., 1 for GPS fix, 2 for DGPS fix)
EARTH_RADIUS_M = 6371000                    # Earth's radius in meters
EARTH_RADIUS_KM = 6371.0                    # Earth's radius in kilometers
KML_NAMESPACE = '{http://www.opengis.net/kml/2.2}'

'''
 Parse all fields from GRMPC line.
 
 Returns:
 - parsed_line (str)
''' 
def parse_GPRMC(fields, parsed_line): 
    time_utc = fields[1]
    status = fields[2]
    if status == "V": # V means invalid data 
        return {}
    latitude_ddmm = float(fields[3]) if fields[3] else None
    lat_dir = fields[4]
    longitude_ddmm = float(fields[5]) if fields[5] else None
    long_dir = fields[6]
    speed = float(fields[7]) * 0.514444  # Convert knots to m/s
    heading = float(fields[8]) if fields[8] else None
    date = fields[9]

    # Convert latitude to decimal degrees
    latitude = None
    if latitude_ddmm is not None:
        degrees = int(latitude_ddmm // 100)
        minutes = latitude_ddmm % 100
        latitude = degrees + (minutes / 60)
        if lat_dir == 'S':  # South latitudes are negative
            latitude = -latitude

    # Convert longitude to decimal degrees
    longitude = None
    if longitude_ddmm is not None:
        degrees = int(longitude_ddmm // 100)
        minutes = longitude_ddmm % 100
        longitude = degrees + (minutes / 60)
        if long_dir == 'W':  # West longitudes are negative
            longitude = -longitude

    try: # Catch faulty dates and skip
        datetime_utc = datetime.strptime(date + time_utc[:6], '%d%m%y%H%M%S')
    except: return {}

    parsed_line["datetime"] = datetime_utc
    parsed_line["status"] = status
    parsed_line["latitude"] = latitude
    parsed_line["longitude"] = longitude
    parsed_line["lat_dir"] = lat_dir
    parsed_line["long_dir"] = long_dir
    parsed_line["speed"] = speed
    parsed_line["heading"] = heading

    return parsed_line 
 
'''
 Parse all fields from GPGGA line.
 
 Returns:
 - parsed_line (str)
'''
def parse_GPGGA(fields, parsed_line): 
    altitude = float(fields[9]) if fields[9] else None
    fix_quality = int(fields[6]) if fields[6] else None
    if altitude is None: # Faulty GPGGA line
        return {}

    parsed_line["altitude"] = altitude
    parsed_line["fix_quality"] = fix_quality
    return parsed_line

'''
 Clean and parses a GPS data line and extract relevant fields.
 
 Returns:
 - final_data (list): all parsed lines 
   each line captured by a dict of all its fields 
'''
def parsed_gps_lines(filename):
    with open(filename, 'r') as file:
        final_data = []

        for line1 in file:
            line1.strip()
            parsed_line  = {}

            if line1.startswith('$GPRMC'):
                line2 = file.readline().strip()

                # Correct case 
                if not '$GPGGA' in line1:
                    fields = line1.split(',')        
                    parsed_line = parse_GPRMC(fields, parsed_line)

                    if line2.startswith('$GPGGA'):
                        # Correct case 
                        if not '$GPRMC' in line2:
                            fields = line2.split(',')        
                            parsed_line = parse_GPGGA(fields, parsed_line)

                        # This happens if arduino eats a nl 
                        elif '$GPRMC' in line2 or 'lng' in line2:
                            fields = line2.split('$')[1].split(',')        # After 1st $ before lng
                            parsed_line = parse_GPGGA(fields, parsed_line)
                            # Do not use lng or next line's GRMPC if present
                    else : 
                        # Means we have no altitude, skip this data point 
                        parsed_line = {}

                # This happens if arduino eats a nl 
                elif '$GPGGA' in line1:
                    fields = line1.split('$')[1].split(',')        # After 1st $
                    parsed_line = parse_GPRMC(fields, parsed_line)

                    fields = line1.split('$')[2].split(',')        # After 2nd $ 
                    parsed_line = parse_GPGGA(fields, parsed_line)
                    # After this jumps to next final condition
            

            # Pretty sure we only care about the altitude here
            elif line1.startswith('$GPGGA'):
                line2 = file.readline().strip()

                # Correct case 
                if not '$GPRMC' in line1:
                    fields = line1.split(',')        
                    parsed_line = parse_GPGGA(fields, parsed_line)

                    if line2.startswith('$GPRMC'):
                        # Correct case 
                        if not '$GPGGA' in line2:
                            fields = line2.split(',')        
                            parsed_line = parse_GPRMC(fields, parsed_line)

                        # This happens if arduino eats a nl 
                        elif ('$GPGGA' in line2) or ('lng' in line2):
                            fields = line2.split('$')[1].split(',')        # After 1st $ before lng
                            parsed_line = parse_GPRMC(fields, parsed_line)
                            # Do not use lng or next line's GRMPC if present
                        
                    else : 
                        # Means we have no GRMPC data, skip this datapoint 
                        parsed_line = {}

                # This happens if arduino eats a nl 
                elif '$GPRMC' in line1:
                    fields = line1.split('$')[1].split(',')        # After 1st $
                    parsed_line = parse_GPGGA(fields, parsed_line)

                    fields = line1.split('$')[2].split(',')        # After 2nd $ 
                    parsed_line = parse_GPRMC(fields, parsed_line)
                    # After this jumps to next final condition
            

            # If no GRMPC/GPGGA line skip this data 
            # Can add in lng line parsing to minimize skipped data
            else: 
                parsed_line = {}

            # skip empty lines 
            if len(parsed_line) > 0 and 'status' in parsed_line: 
                final_data.append(parsed_line)
        
        return final_data
            
"""
Reports date and time that trip occured in YYYY/MM/DD and UTC format

Returns:
- date
- time 
"""
def trip_date_occurance(parsed_data):
    date  = parsed_data[0]["datetime"].date().strftime('%Y/%m/%d')
    time  = parsed_data[0]["datetime"].time()
    return date, time

"""
Checks if a coordinate is within a given radius of a target location.

Parameters:
    coord (tuple): Tuple of (latitude, longitude) for the trip point.
    target (tuple): Tuple of (latitude, longitude) for the target location.
    radius (float): Distance in kilometers to consider 'near'. Default is 0.5 km.
"""
def is_near_location(coord, target, radius=0.5):
    # Haversine formula
    EARTH_RADIUS_KM = 6371.0  
    lat1, lon1 = radians(coord[0]), radians(coord[1])
    lat2, lon2 = radians(target[0]), radians(target[1])
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    distance = EARTH_RADIUS_KM * c

    return distance <= radius
 
"""
Determines if the trip started near a given location.
"""
def trip_started_near_location(trip_data, target_location, radius=2):
    start_point = (trip_data[0]['latitude'], trip_data[0]['longitude'])
    return is_near_location(start_point, target_location, radius)

"""
Determines if the trip ended near a given location.
"""
def trip_ended_near_location(trip_data, location_b, radius=2):
    lat = trip_data[-1]['latitude']
    long = trip_data[-1]['longitude']
    end_point = (trip_data[-1]['latitude'], trip_data[-1]['longitude'])
    return is_near_location(end_point, location_b, radius)


"""
Check if the trip is considered a full trip.

Parameters:
- trip_data: List of parsed GPS data points (dict).

To be a full trip:
- The GPS device must have a location lock ("A") before the trip
  started and after the trip ended.
- No sudden jumps in location greater than a threshold.

Returns:
- True if valid, False if invalid
"""
def check_if_full_trip(trip_data):
    # Get the start and last data points 
    start_point = trip_data[0]
    end_point = trip_data[-1]

    # Check if GPS location lock exists
    if start_point["status"] != "A" or end_point["status"] != "A":
        print('No,')  
        return False

    # Find distances between consecutive points and check for sudden jumps
    for i in range(1, len(trip_data)):
        current_point = trip_data[i]
        previous_point = trip_data[i - 1]

        # Find distance between consecutive points
        distance = geodesic(
            (previous_point["latitude"], previous_point["longitude"]),
            (current_point["latitude"], current_point["longitude"])
        ).meters

        # Check if there was a sudden jump
        if distance > JUMP_THRESHOLD:
            return False

    # If no issues were found, the trip is valid
    return True


'''
 Computes total time between first and last points.
 
 Parameters:
- trip_data: List of parsed GPS data points (dict).

 Returns:
 - duration (timedelta): Amount of time the trip took
'''
def compute_trip_duration(trip_data):
    # Find the first point where the device starts moving
    start_index = None
    for i, point in enumerate(trip_data):
        if point["speed"] >= MIN_SPEED_THRESHOLD:
            start_index = i
            break

    # Check if valid starting point
    if start_index is None:
        # Device was stationary the entire time on
        return None 

    # Find the last point where the device is moving before stopping at destination
    end_index = None
    for i in range(len(trip_data) - 1, -1, -1):
        if trip_data[i]["speed"] >= MIN_SPEED_THRESHOLD:
            end_index = i
            break

    # Check if valid ending point
    if end_index is None:
        # Device stopped recording before arriving at destination
        return None  

    # Find the trip duration (start to end, excluding stationary periods)
    start_time = trip_data[start_index]["datetime"]
    end_time = trip_data[end_index]["datetime"]

    duration = end_time - start_time
    return duration

"""
Computes the total amount of stops that occur during the trip
based on how long the device is stationary and or 
moving less than 5 mph

Parameters:
- trip_data: List of parsed GPS data points (dict).

Returns:
- total_stops (int): Total stops during the trip.
"""
def how_many_stops(trip_data):       
    total_stops = 0
    stop_start_time = None

    for i in range(1, len(trip_data)):
        
        current_point = trip_data[i]

        # If the speed is below 5 mph, potentially stop
        if current_point["speed"] <= STOP_SPEED:
            
            if stop_start_time is None:
                # Get the start of a potential stop
                stop_start_time = current_point["datetime"]
                
        else:
            # If the vehicle is moving, check if we were stopped
            if stop_start_time is not None:
                # Find the duration of the stop
                stop_duration = current_point["datetime"] - stop_start_time
                
                if MIN_STOP_DURATION <= stop_duration <= MAX_STOP_DURATION:
                    total_stops += 1
                    
                # Reset stop tracker
                stop_start_time = None

    return total_stops

'''
Computes the fraction of the trip spent going uphill (angle > 15 degrees).

Parameters:
- total_time: Total time of trip. 
- trip_data: List of parsed GPS data points (dict).

Returns:
- total uphill time (int)
'''
def compute_uphill_duration(total_time, trip_data): 
    if total_time.total_seconds() == 0:
        return 0.0  # Avoid division by zero for empty trips.

    uphill_time = 0  
    prev_point = None

    for point in trip_data:
        if not prev_point:
            prev_point = point
            continue

        # Calculate the changes in altitude and distance
        altitude_change = point['altitude'] - prev_point['altitude']
        lat1, long1 = prev_point['latitude'], prev_point['longitude']
        lat2, long2 = point['latitude'], point['longitude']
        
        # Compute horizontal distance between consecutive points using haversine formula
        horizontal_distance = haversine(lat1, long1, lat2, long2)  
        
        if horizontal_distance > 0:  # Avoid division by zero
            angle = degrees(atan(altitude_change / horizontal_distance))
            if angle > 15:
                # Compute time spent between these two points
                time_difference = (point['datetime'] - prev_point['datetime']).total_seconds()
                uphill_time += time_difference

        prev_point = point

    # Calculate percentage of time spent going uphill
    uphill_fraction = (uphill_time / total_time.total_seconds()) * 100

    # HH:MM::SS format
    uphill_duration = str(timedelta(seconds=uphill_time))
    return round(uphill_fraction, 1), uphill_duration


"""
Calculates distance between two points on the Earth's surface.

Parameters:
- lat1, lon1: Latitude and longitude of the first point in decimal degrees.
- lat2, lon2: Latitude and longitude of the second point in decimal degrees.

Returns:
- distance found (float)
"""
def haversine(lat1, lon1, lat2, lon2):

    from math import radians, sin, cos, sqrt, atan2
    EARTH_RADIUS_M = 6371000  

    # Convert coordinates from degrees to radians
    phi1, phi2 = radians(lat1), radians(lat2)
    delta_phi = radians(lat2 - lat1)
    delta_lambda = radians(lon2 - lon1)

    # Haversine formula
    a = sin(delta_phi / 2) ** 2 + cos(phi1) * cos(phi2) * sin(delta_lambda / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    return EARTH_RADIUS_M * c


"""
Computes the total climb for each hill during a trip.

Parameters:
- trip_data : List of parsed GPS data points (dict).

args: flat_threshold: Horizontal distance in m defining max flat section that one hill can span

Returns:
- total_climb (float): Total meters climbed during the trip.
- hills (list): List of individual hill climbs.
"""
def compute_hill_climbs(trip_data, flat_threshold=50):
    total_climb = 0
    hills = []
    current_hill_climb = 0
    flat_distance_accumulated = 0  # To track flat sections
    prev_point = None

    for point in trip_data:
        if not prev_point:
            prev_point = point
            continue

        # Calculate changes in altitude and horizontal distance
        altitude_change = point['altitude'] - prev_point['altitude']
        lat1, long1 = prev_point['latitude'], prev_point['longitude']
        lat2, long2 = point['latitude'], point['longitude']
        horizontal_distance = haversine(lat1, long1, lat2, long2)

        if altitude_change > 0:    # Continue climbing
            current_hill_climb += altitude_change
            flat_distance_accumulated = 0  # Reset flat section tracker


        elif altitude_change <= 0:  # Possible end of a hill
            if horizontal_distance > flat_threshold: #too flat for too long 
                if current_hill_climb > 0:
                    hills.append(round(current_hill_climb,1))
                    total_climb += current_hill_climb
                    current_hill_climb = 0  # Reset for next hill
                flat_distance_accumulated = 0  # Reset tracker
            else:
                # Flat section, accumulate distance
                flat_distance_accumulated += horizontal_distance
                if flat_distance_accumulated > flat_threshold:
                    # End the hill if the flat section becomes too long
                    if current_hill_climb > 0:
                        hills.append(round(current_hill_climb,1))
                        total_climb += current_hill_climb
                        current_hill_climb = 0  # Reset for next hill
                    flat_distance_accumulated = 0  # Reset tracker
                else: 
                    current_hill_climb += altitude_change  # Continue climbing

        prev_point = point

    # Account for a hill still in progress at the end of the trip
    if current_hill_climb > 0:
        hills.append(round(current_hill_climb,1))
        total_climb += current_hill_climb

    return round(total_climb,1), hills

"""
Computes the amount of brake 'slams' that occur during the trip.

Parameters:
- trip_data : List of parsed GPS data points (dict).

Returns:
- brake_count (int): Total amount of brake 'slams'.
"""
def check_brake_rate(trip_data):
    brake_counter = 0
    total_deceleration = 0.0
    
    for i in range(1, len(trip_data)):
        current_point = trip_data[i]
        previous_point = trip_data[i - 1]
        
        # skip if speed is 0
        if current_point["speed"] == 0 or previous_point["speed"] == 0:
            continue
        
        # Convert speed from knots to meters per second
        current_speed_mps = current_point["speed"] * 0.514444
        previous_speed_mps = previous_point["speed"] * 0.514444
        
        # Check if we're decelerating (current speed is less than previous speed)
        if current_speed_mps < previous_speed_mps:
            time_diff = (current_point["datetime"] - previous_point["datetime"]).total_seconds()
            speed_diff = previous_speed_mps - current_speed_mps
            
            # Compute speed/time ratio (deceleration in m/s²)
            if time_diff > 0:  # Avoid division by zero
                deceleration = speed_diff / time_diff
                
                # Add to total deceleration
                total_deceleration += deceleration
                
                # Check if the next point is also decelerating
                if i + 1 < len(trip_data):
                    next_point = trip_data[i + 1]
                    next_speed_mps = next_point["speed"] * 0.514444
                    
                    if next_speed_mps < current_speed_mps:
                        # Continue the deceleration sequence
                        continue
                    else:
                        # Acceleration detected, stop the deceleration sequence
                        if deceleration > BRAKE_THRESHOLD:
                            brake_counter += 1
                            
                        total_deceleration = 0.0  # Reset the deceleration total after the event
                else:
                    # Check if the last point also exceeds the threshold
                    if deceleration > BRAKE_THRESHOLD:
                        brake_counter += 1
                        
    return brake_counter

"""
 Generates a KML file for the given GPS trip data.
    
Parameters:
- trip_data: List of parsed GPS data points (dict).
- output_filename: The filename for the generated KML file.
"""
def generate_kml(trip_data, output_file="output_as_kml.kml"):
   
    document = kml.KML()
    kml_document = kml.Document(ns=KML_NAMESPACE)
    document.append(kml_document)

    coordinates = []
    for point in trip_data:
        if point.get("latitude") and point.get("longitude"):
            # Add each point's (long, lat, alt) as a coordinate to the line
            coordinates.append((point["longitude"], point["latitude"], point["altitude"]))

    line_geometry = LineString(coordinates)
    
    # Make the line pretty
    line_style = LineStyle(KML_NAMESPACE, id="lineStyleExample")
    line_style.color = "ffff0000"
    line_style.width = 5 
    
    style = Style(KML_NAMESPACE, id="Style")
    style.append_style(line_style)

    # Make Placemark and assign the geometry and style
    placemark = kml.Placemark(ns=KML_NAMESPACE)
    placemark.name = "GPS Trip Route"
    placemark.geometry = line_geometry
    placemark.styleUrl = "Style"

    kml_document.append_style(style)
    kml_document.append(placemark)

    # Add Start Point
    start_point = trip_data[0]
    start_placemark = kml.Placemark()
    start_placemark.name = "Start"
    start_placemark.geometry = Point(start_point["longitude"], start_point["latitude"], start_point["altitude"])
    
    kml_document.append(start_placemark)

    # Add End Point
    end_point = trip_data[-1]
    end_placemark = kml.Placemark()
    end_placemark.name = "End"
    end_placemark.geometry = Point(end_point["longitude"], end_point["latitude"], end_point["altitude"])
    
    kml_document.append(end_placemark)
    
    # Write to the output file
    with open(output_file, "w") as kml_file:
        kml_file.write(document.to_string(prettyprint=True))


# Main subroutine navigating all questions 
# Full analyzes a given GPS trip 
def main():
    filename = sys.argv[1]
    trip_data = parsed_gps_lines(filename)
    # Q1-2 
    date, time = trip_date_occurance(trip_data)

    # Q3-6 
    RIT_location = (43.086065, -77.68094333)
    kinsman_res_location = (43.138238, -77.437821)

    trip_start_near_drk = lambda: "Yes" if trip_started_near_location(trip_data, kinsman_res_location, 0.28) else "No"    

    trip_end_near_rit = lambda: "Yes" if trip_ended_near_location(trip_data, RIT_location, 0.93) else "No"

    # Tighter bound for 'start AT' location
    trip_start_at_rit = lambda: "Yes" if trip_started_near_location(trip_data, RIT_location, 0.8) else "No"    

    # Tighter bound for 'start AT' location, within 30m 
    trip_end_near_drk = lambda: "Yes" if trip_ended_near_location(trip_data, kinsman_res_location, 0.03) else "No"
    
    # Q7
    was_full_trip = lambda: "Yes" if check_if_full_trip(trip_data) else "No"

    # Q8
    duration = compute_trip_duration(trip_data)

    # Q9
    stops = how_many_stops(trip_data)
    
    # Q10-11
    uphill_percent, uphill_duration = compute_uphill_duration(duration, trip_data)

    # Q12
    total_climb, hills = compute_hill_climbs(trip_data)

    # Q13
    num_of_brakes = check_brake_rate(trip_data)
    
    # Q14-15
    generate_kml(trip_data)
    
    # Q/A pretty-printing: 
    questions_and_answers = [
        ("GPS Trip Occured on:", date),
        ("Trip started at time:", time),
        ("Did the trip originiate at Dr. K's house?", trip_start_near_drk()),
        ("Did the trip go to RIT?", trip_end_near_rit()),
        ("Did the trip originate at RIT?", trip_start_at_rit()),
        ("Did the trip go to Dr. K's House?", trip_end_near_drk()),
        ("Was the trip a full trip?", was_full_trip()),
        ("Trip Duration:", duration),
        ("Number of complete stops:", stops),
        ("Percent of time spent traveling uphill", uphill_percent),
        ("Time spent traveling uphill:", uphill_duration),
        ("Total hills climbed:", len(hills)), 
    ]

    # Print the questions and answers with proper formatting
    for question, answer in questions_and_answers:
        print(f"{question:<45} | {answer}")

    hill_count = 1
    for hill in hills: 
        print(f" * hill {hill_count} - climbed {hill}m")
        hill_count +=1
    
    questions_and_answers = [
        ("Total meters climbed uphill:", total_climb),
        ("Times breaks were 'slammed':", num_of_brakes)
    ]
    for question, answer in questions_and_answers:
        print(f"{question:<45} | {answer}")

main()
