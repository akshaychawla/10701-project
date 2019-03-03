import os


def get_filenames(path):
    categories = sorted(os.listdir(path))
    y = []
    x = []
    for label in categories:
        for f in os.listdir(path + "/" + label):
            fullpath = os.path.join(path + "/" + label, f)
            x.append(fullpath)
            y.append(label)
    return x, y

# if(__name__ == '__main__'):
#     x, y = get_filenames(BASE_PATH)
#     X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.2)
    # print(len(X_train), len(X_test), len(y_train), len(y_test))
    # print(X_train[0:5], y_train[0:5])
    # print(len(set(y_train)), len(set(y_test)))
    

    