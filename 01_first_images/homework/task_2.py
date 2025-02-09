import cv2
import numpy as np

def find_road_number(image: np.ndarray) -> int:
    """
    Найти номер дороги, на которой нет препятсвия в конце пути.

    :param image: исходное изображение
    :return: номер дороги, на котором нет препятсвия на дороге
    """
    road_number = None
    
    image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    
    hsv_lane = [(25, 100, 100), (35,255,255)]
    hsv_obstacle = [(0, 100, 100), (10,255,255)]
    hsv_car = [(110, 100, 100), (130,255,255)]

    lane_i = cv2.inRange(image, hsv_lane[0], hsv_lane[1])
    obstacle_i = cv2.inRange(image, hsv_obstacle[0], hsv_obstacle[1])
    car_i = cv2.inRange(image, hsv_car[0], hsv_car[1])
    
    f_row = lane_i[0]
    lanes = []
    
    i = 0
    while (i < image.shape[1]):
        if (f_row[i] == 0):
            start = i
            while (f_row[i] == 0):
                i += 1
            lanes.append([start, i])
        i+=1

    obstacles = []
    car_lane = -1
    for i, l in enumerate(lanes):
        obstacle = np.any(obstacle_i[:, l[0]:l[1]] == 255)
        obstacles.append(obstacle)
        if (np.any(car_i[:,l[0]:l[1]]==255)):
            car_lane = i 

    return obstacles.index(False)