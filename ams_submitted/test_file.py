import matplotlib.pyplot as plt
import time

def get_input():
    x = [1,2,4]
    y = [4, 7, 9]
    plt.plot(x,y)
    plt.show()

def count_to_num():
    for i in range(5):
        time.sleep(1)
        print(i)