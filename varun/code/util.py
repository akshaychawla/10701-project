import os
import matplotlib.pyplot as plt

def save_wordmap(wordmap, filename):
	fig = plt.figure(2)
	plt.axis('equal')
	plt.axis('off')
	plt.imshow(wordmap, cmap="rainbow")
	plt.savefig(filename, pad_inches=0)

def display_filter_responses(response_maps):
	'''
	Visualizes the filter response maps.

	[input]
	* response_maps: a numpy.ndarray of shape (H,W,3F)
	'''
	
	fig = plt.figure(1)
	
	for i in range(20):
		plt.subplot(5,4,i+1)
		resp = response_maps[:,:,i*3:i*3+3]
		resp_min = resp.min(axis=(0,1),keepdims=True)
		resp_max = resp.max(axis=(0,1),keepdims=True)
		resp = (resp-resp_min)/(resp_max-resp_min)
		plt.imshow(resp)
		plt.axis("off")

	plt.subplots_adjust(left=0.05,right=0.95,top=0.95,bottom=0.05,wspace=0.05,hspace=0.05)
	plt.show()

def get_filenames(path):
    categories = sorted(os.listdir(path))
    y = []
    x = []
    for i,label in enumerate(categories):
        for f in os.listdir(path + "/" + label):
            fullpath = os.path.join(path + "/" + label, f)
            x.append(fullpath)
            y.append(i)
    return x, y

# if(__name__ == '__main__'):
#     x, y = get_filenames(BASE_PATH)
#     X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.2)
    # print(len(X_train), len(X_test), len(y_train), len(y_test))
    # print(X_train[0:5], y_train[0:5])
    # print(len(set(y_train)), len(set(y_test)))
    

    