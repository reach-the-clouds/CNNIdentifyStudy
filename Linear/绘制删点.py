import matplotlib.pyplot as plt
import numpy as np

data = [[0.8, 1.0], [1.7, 0.9], [2.7, 2.4], [3.2, 2.9], [3.7, 2.8], [4.2, 3.8], [4.2, 2.7]]
data_np = np.array(data)
data_x = data_np[:, 0]
data_y = data_np[:, 1]

plt.scatter(data_x, data_y, c='r', linewidths=1)
plt.xlim([0, 8])
plt.ylim([0, 8])
plt.xlabel("x")
plt.ylabel("y")

w = 0
x_0 = 0
x_8 = 8
y_0 = x_0 * w
y_8 = x_8 * w
x_ = data_x
y_ = x_ * 0.1
line, = plt.plot([x_0, x_8], [y_0, y_8], 'r-')

m = np.linspace(0, 2, 100)
for w in m:
    y_0 = x_0 * w
    y_8 = x_8 * w
    line.set_data([x_0, x_8], [y_0, y_8])
    plt.pause(1)
plt.show()
