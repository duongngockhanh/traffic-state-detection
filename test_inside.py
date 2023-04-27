from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

def isInside(points, centroid):
    polygon = Polygon(points)
    centroid = Point(centroid)
    return polygon.contains(centroid)

points = [(693, 262), (1161, 255), (1728, 914), (246, 830)]
centroid = [170.73834228515625, 188.60047912597656]

print(isInside(points, centroid))