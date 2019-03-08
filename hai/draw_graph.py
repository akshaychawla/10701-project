import os
import sys
import numpy as np
import matplotlib.pyplot as plt

def main():
    erroes = np.array([0.1608433734939759 , 0.1578313253012048, 
            0.15843373493975904, 0.1578313253012048, 0.15843373493975904])
    y_test = 1.0 - erroes
    y_Train = np.array([1.0,1.0,1.0,1.0,1.0])
    print(y_test)
    c = [0.1, 1.0, 10, 20, 100]
    plt.figure(1)
    plt.plot(y_test, linestyle='-' ,color='r' ,marker='o', label='Test accuracy',lw=2)
    plt.plot(y_Train, linestyle='-' ,color='b' ,marker='o', label='Train accuracy',lw=2)
    plt.xticks(np.arange(5), ("0.1","1", "10", "20", "100"))
    plt.ylabel('Acc')
    plt.xlabel('C')
    plt.legend()
    # plt.title('SVM classification for Caltect')
    plt.show()
if __name__ == '__main__':
	main()
