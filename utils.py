import mat73
def convertMat2Png():
    path2dataset = "nyu_depth_v2_labeled.mat"
    images_and_labels = mat73.loadmat(path2dataset)

    for i in range(images_and_labels['images'].shape[3]):
        plt.imsave('Datas/imgNo{0}.png'.format(i), images_and_labels['images'][:,:,:,i])
        plt.imsave('Datas/depthNo{0}.png'.format(i), images_and_labels['depths'][:,:,i])
        plt.imsave('Datas/labelNo{0}.png'.format(i), images_and_labels['labels'][:,:,i])
