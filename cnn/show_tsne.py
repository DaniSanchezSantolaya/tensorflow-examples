import numpy as np
import pylab as plt
import os
from sklearn.manifold import TSNE


#idx = np.arange(10000)
#np.random.shuffle(idx)
#idx = idx[:1000]

mode = 'plot'
#layers = ['fc1','fc2','flatten']
#features_path = './features/task1/reg'
layers = ['L2-norm']
features_path = './features/task2'
labels_file = os.path.join('./features', 'labels_test.npy')
labels = np.load(labels_file)
#labels = labels[idx]
print labels.shape
labels = np.argmax(labels, axis=1)

for layer in layers:
    features_file = os.path.join(features_path, 'features_' + str(layer) + '_test.npy')
    print 'load featurres from: ' + str(features_file)
    features = np.load(features_file)
    #features = features[idx]
    print features.shape


    model = TSNE(n_components=2, random_state=0)
    Y = model.fit_transform(features) 
    print Y.shape

    if mode == 'save':
        file_name = 'tsne/task2/tsn_' + str(layer) + '_reg.npy'
        np.save(file_name, Y)
    elif mode == 'plot':
        plt.scatter(Y[:, 0], Y[:,1], 10, labels)
        plt.title('t-SNE representation for the layer ' + str(layer))
        plot_file = 'report/task2_tsn_' + str(layer) + '_reg.png'
        plt.savefig(plot_file)
        #plt.show()



