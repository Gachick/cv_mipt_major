import cv2
import numpy as np


def rotate(image, point: tuple, angle: float) -> np.ndarray:
    """
    Повернуть изображение по часовой стрелке на угол от 0 до 360 градусов и преобразовать размер изображения.

    :param image: исходное изображение
    :param point: значение точки (x, y), вокруг которой повернуть изображение
    :param angle: угол поворота
    :return: повернутное изображение
    """
    
    RM = cv2.getRotationMatrix2D(point, angle, scale=1.0)

    rows, cols, _ = image.shape
    rows -= 1
    cols -= 1

    angles = np.array([[0, cols, cols, 0],[0, 0, rows, rows], [1,1,1,1]])
    new_angles = RM @ angles

    n_rows = int(np.max(new_angles[0])-np.min(new_angles[0]))
    n_cols = int(np.max(new_angles[1])-np.min(new_angles[1]))

    x_shift = -int(np.min(new_angles[0]))
    y_shift = -int(np.min(new_angles[1]))

    RM[0, 2] += x_shift
    RM[1, 2] += y_shift

    image = cv2.warpAffine(image.copy(), RM, (n_rows,n_cols))

    return image


def apply_warpAffine(image, points1, points2) -> np.ndarray:
    """
    Применить афинное преобразование согласно переходу точек points1 -> points2 и
    преобразовать размер изображения.

    :param image:
    :param points1:
    :param points2:
    :return: преобразованное изображение
    """
    points1=np.array(points1, np.float32)
    points2=np.array(points2, np.float32)

    PM = cv2.getPerspectiveTransform(points1, points2) 

    rows, cols, _ = image.shape
    rows -= 1
    cols -= 1

    angles = np.array([[0, cols, cols, 0],[0, 0, rows, rows], [1,1,1,1]])
    new_angles = PM @ angles
    new_angles /= new_angles[2]


    n_rows = int(np.max(new_angles[0])-np.min(new_angles[0]))
    n_cols = int(np.max(new_angles[1])-np.min(new_angles[1]))

    x_shift = -int(np.min(new_angles[0]))
    y_shift = -int(np.min(new_angles[1]))

    shift = np.array([x_shift, y_shift], np.float32)

    PM = cv2.getPerspectiveTransform(points1, points2+shift)
    image = cv2.warpPerspective(image.copy(), PM, (n_rows,n_cols))

    return image

def normalize_document(image) -> np.ndarray:
    def offset(hsv, diff):
        hsv =hsv.astype(int)
        hsv += diff
        hsv = np.clip(hsv, [0, 0, 0], [180, 255, 255])
        return hsv.astype(np.uint8)

    rows, cols, _ = image.shape
    image_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    center = image_hsv[rows//2, cols//2]

    mask = cv2.inRange(image_hsv, offset(center, [-5, -100, -100]), offset(center, [5, 100, 100]))
    # fig, axs = plt.subplots(1, 1, figsize = (8, 8))
    # axs.imshow(mask, cmap='grey')
    # axs.scatter([top[0], left[0], right[0], bottom[0]],[top[1], left[1], right[1], bottom[1]])

    x_proj = cv2.reduce(mask, 0, cv2.REDUCE_MAX)
    y_proj = cv2.reduce(mask, 1, cv2.REDUCE_MAX)

    left, right = -1, -1
    for i, v in enumerate(x_proj.flatten()):
        left = i if left == -1 and v else left
        right = i if v else right

    top, bottom = -1, -1
    for i, v in enumerate(y_proj.flatten()):
        top = i if top == -1 and v else top
        bottom = i if v else bottom

    left = np.array([left, np.argmax(mask[:,left])])
    right = np.array([right, np.argmax(mask[:,right])])
    top = np.array([np.argmax(mask[top,:]), top])
    bottom = np.array([np.argmax(mask[bottom,:]), bottom])

    width = int(np.linalg.norm(left - bottom))
    height = int(np.linalg.norm(top - left))


    PM = cv2.getPerspectiveTransform(
        np.array([top, left, right, bottom], np.float32),
        np.array([[0,0], [0, height], [width, 0], [width, height]], np.float32)
    )
    res = cv2.warpPerspective(image.copy(), PM, (width,height))
    
    return res