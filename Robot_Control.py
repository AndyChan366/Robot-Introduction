import sim
import time
import numpy as np
import cv2
import math


# 获取传感器采集的路径相关信息
def GetInfo(clientId):
    errorCode, visionSensorHandle = sim.simxGetObjectHandle(clientId, 'Vision_sensor',
                                                            sim.simx_opmode_oneshot_wait)  # 获取相机（传感器）句柄
    # 第一次获取数据无效
    errprCode, resolution, image = sim.simxGetVisionSensorImage(clientId, visionSensorHandle, 0,
                                                                sim.simx_opmode_streaming)  # 获取图像
    _, pos = sim.simxGetObjectPosition(clientId, visionSensorHandle, -1, sim.simx_opmode_streaming)
    time.sleep(0.1)
    while True:  # 循环获取图像数据
        errprCode, resolution, image = sim.simxGetVisionSensorImage(clientId, visionSensorHandle, 0,
                                                                    sim.simx_opmode_buffer)
        sensorImage = np.array(image, dtype=np.uint8)
        sensorImage.resize([resolution[1], resolution[0], 3])  # 480行 640列
        grayImage = cv2.cvtColor(sensorImage, cv2.COLOR_BGR2GRAY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        eroded = cv2.erode(grayImage, kernel)
        dst = cv2.medianBlur(eroded, 5)
        cv2.flip(dst, 0, dst)
        cv2.imshow('result', dst)  # 显示图像
        direction, angle = pro(dst)
        print(direction, angle)
        v_left = 1 + direction * angle * angle / 8100  # 设定左右速度
        v_right = 1 - direction * angle * angle / 8100

        _, l_motor = sim.simxGetObjectHandle(clientId, "Pioneer_p3dx_leftMotor",
                                             sim.simx_opmode_oneshot_wait)  # 获取轮子控制句柄
        _, r_motor = sim.simxGetObjectHandle(clientId, "Pioneer_p3dx_rightMotor", sim.simx_opmode_oneshot_wait)
        errorCode = sim.simxSetJointTargetVelocity(clientId, l_motor, 1.4 * v_left, sim.simx_opmode_oneshot)  # 发送指令给左右轮子，控制速度，这里不等待返回结果，减小控制命令间隔，增大控制精度
        errorCode = sim.simxSetJointTargetVelocity(clientId, r_motor, 1.4 * v_right, sim.simx_opmode_oneshot)
        if cv2.waitKey(1) == 27:  # 可以等待ESC指令退出
            break
    return dst


# 处理图像数据来获取控制的信息
def pro(result):
    row = 420        # 第一个采样行
    high = 240       # 第二个采样行
    mid = 320        # 中间列
    endpoint_high = 320   # 中间列
    endpoint_left1 = 0    # 左侧第一个黑色像素点（黑白像素交界位置，即路径最左侧）
    endpoint_right1 = 0   # 距离中心线最近的黑色像素点（左）
    endpoint_left2 = 640  # 距离中心线最近的黑色像素点（右）
    endpoint_right2 = 640 # 右侧第一个黑色像素点（黑白像素交界位置，即路径最右侧）
    endpoint = 320       # 找到的路径的中间位置
    direction = 0        # 方向（-1左，1右，0直行）
    for j in range(mid, 1, -1):  # 对于第420行，循环变历检查图像中路径（黑色像素点）位置
        if (result[row, j] < np.array([30, 30, 30])).all() and endpoint_right1 < j:
            endpoint_right1 = j
            for jj in range(j, 1, -1):
                if (result[row, jj] > np.array([30, 30, 30])).all() and endpoint_left1 < jj:
                    endpoint_left1 = jj
                    break
            break
    for j in range(mid, 640, 1): # 同上
        if (result[row, j] < np.array([30, 30, 30])).all() and endpoint_left2 > j:
            endpoint_left2 = j
            for jj in range(j, 640, 1):
                if (result[row, jj] > np.array([30, 30, 30])).all() and endpoint_right2 > jj:
                    endpoint_right2 = jj
                    break
            break
    print(endpoint_left1, endpoint_right1, endpoint_left2, endpoint_right2)
    if endpoint_right1 + endpoint_left2 > mid * 2:
        endpoint = (endpoint_left1 + endpoint_right1) / 2
        direction = -1  # left
    elif endpoint_right1 + endpoint_left2 < mid * 2:
        endpoint = (endpoint_left2 + endpoint_right2) / 2
        direction = 1   # right
    else:
        endpoint = mid
        direction = 0   # mid
    if endpoint != 320: # 此时不在中心线上，算角度
        angle = cal_ang(np.array([row, mid]), np.array([480, mid]), np.array([row, endpoint]))
    else:
        angle = 0       # 此时已经在路径中心线上，偏离角度为0
    # 下面的代码为对于第二个采样行进行分析，由于第二个采样行只是起到辅助作用解决一些特殊情况，所以在扫路径的时候只扫一个方向，而不是对图像所有列进行遍历
    # 只考虑左侧
    if (direction == -1):
        endpoint_left = 0
        endpoint_right = 0
        for j in range(mid, 1, -1):
            if ((result[high, j] < np.array([10, 10, 10])).all() and endpoint_right < j):
                endpoint_right = j
                for jj in range(j, 1, -1):
                    if ((result[high, jj] > np.array([10, 10, 10])).all() and endpoint_left < jj):
                        endpoint_left = jj
                        break
                break
    # 只考虑右侧
    else:
        endpoint_left = 0
        endpoint_right = 0
        for j in range(mid, 640):
            if (result[high, j] < np.array([10, 10, 10])).all() and endpoint_left > j:
                endpoint_left = j
                for jj in range(j, 640):
                    if (result[high, jj] > np.array([10, 10, 10])).all() and endpoint_right > jj:
                        endpoint_right = jj
                        break
                break
    # 同上
    endpoint_high = (endpoint_left + endpoint_right) / 2
    if endpoint_high != 320:
        angle_high = cal_ang(np.array([high, mid]), np.array([480, mid]), np.array([high, endpoint_high]))
    else:
        angle_high = 0
    # 差速修正系数
    if angle != 0:
        direction *= 5 * (angle_high / angle)
    return direction, angle


# 用于计算偏差角度
def cal_ang(point_1, point_2, point_3):
    """
    根据三点坐标计算夹角
    :param point_1: 点1坐标
    :param point_2: 点2坐标
    :param point_3: 点3坐标
    :return: 返回任意角的夹角值，这里只是返回点2的夹角
    """
    a = math.sqrt(
        (point_2[0] - point_3[0]) * (point_2[0] - point_3[0]) + (point_2[1] - point_3[1]) * (point_2[1] - point_3[1]))
    b = math.sqrt(
        (point_1[0] - point_3[0]) * (point_1[0] - point_3[0]) + (point_1[1] - point_3[1]) * (point_1[1] - point_3[1]))
    c = math.sqrt(
        (point_1[0] - point_2[0]) * (point_1[0] - point_2[0]) + (point_1[1] - point_2[1]) * (point_1[1] - point_2[1]))
    A = math.degrees(math.acos((a * a - b * b - c * c) / (-2 * b * c)))
    B = math.degrees(math.acos((b * b - a * a - c * c) / (-2 * a * c)))
    C = math.degrees(math.acos((c * c - a * a - b * b) / (-2 * a * b)))
    return B


# 连接v-rep
def connection(num):
    print('Program Start!')
    sim.simxFinish(-1)  # 关掉之前的连接
    clientId = sim.simxStart("127.0.0.1", 19999, True, True, 5000, 5)  # 建立和服务器的连接
    wrong_nums = 100
    while True:
        if clientId != -1:  # 连接成功
            print('Successful Connect!')
            GetInfo(clientId)  # 获取传输的图像
            wrong_nums = 100
            break
        else:
            time.sleep(0.2)
            clientId = sim.simxStart("127.0.0.1", 19999, True, True, 5000, 5)  # 重新建立和服务器的连接
            wrong_nums = wrong_nums - 1
        if wrong_nums == 0:
            print('Fail Connect!')  # 尝试100次未果则不再连接
            break
    sim.simxFinish(clientId)  # 结束连接
    print('Program Ended!')


if __name__ == "__main__":
    connection(0)
