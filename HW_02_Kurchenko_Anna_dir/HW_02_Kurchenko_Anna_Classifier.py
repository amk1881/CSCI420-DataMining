
def classify_driver(speed, threshold):
    """
    Classifies a driver based on their speed and the given threshold.

    Parameters:
    - speed (float): The speed of the driver in miles per hour.
    - threshold (float): The speed threshold used for classification.

    Returns:
    - int: The intent of the driver (0 for safe, 1 for normal, 2 for aggressive).
    """
    if speed <= threshold:
        intent = 0
    else:
        intent = 2 
    return intent
