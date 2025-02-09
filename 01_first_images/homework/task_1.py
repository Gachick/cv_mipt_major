import numpy as np
import cv2

def find_way_from_maze(image: np.ndarray) -> tuple:
    """
    Найти путь через лабиринт.

    :param image: изображение лабиринта
    :return: координаты пути из лабиринта в виде (x, y), где x и y - это массивы координат
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    f_row = image[0]
    x_entry = np.argmax(f_row > 0)
    
    queue = []
    queue.append([0, x_entry])

    rows, cols = image.shape
    parents = np.full((rows, cols, 2), -1)
    parents[0, x_entry, 0] = -2
    parents[0, x_entry, 1] = -2

    last = []
    while(len(queue) != 0):
        cur = queue.pop(0)
        if cur[0] == rows-1:
            last = cur
            break
        for offset in [[-1, 0], [1, 0], [0, 1], [0, -1]]:
            c_row = np.clip(cur[0] + offset[0], 0, rows-1)
            c_col = np.clip(cur[1] + offset[1], 0, cols-1)
            if parents[c_row, c_col,0]==-1 and parents[c_row, c_col, 1]==-1 and image[c_row, c_col] == 255:
                parents[c_row, c_col, 0] = cur[0]
                parents[c_row, c_col, 1] = cur[1]
                queue.append([c_row, c_col])
    x = []
    y = []
    cur = last
    while(parents[cur[0], cur[1],0]!=-2):
        x.append(cur[0])
        y.append(cur[1])
        cur = [parents[cur[0], cur[1],0], parents[cur[0], cur[1],1]]
    x.append(0)
    y.append(x_entry)

    return (x, y)