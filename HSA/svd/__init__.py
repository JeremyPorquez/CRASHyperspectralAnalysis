import numpy as np
from multiprocessing.pool import ThreadPool

def anscombe(data):
    return 2 * np.sqrt(data + 3. / 8)

def inverse_anscombe(data):
    return (data / 2.) ** 2 - 3. / 8

def svd_filter(data, full_matrices=False, singular_values=10, full_result=False):
    '''

    :param data:            3D numpy matrix with shape [spectra, y, x]

    :param full_matrices:   bool, optional
                            performs full SVD or reduced SVD

    :param singular_values: int, optional
                            number of first singular values to retain

    :param full_result:     bool, optional
                            returns SVD filtered image, U, S, V
                            else returns SVD filtered image

    :return:
                            SVD filtered image (optional: U, S, V with full_result = True)
    '''



    if not type(data) == np.ndarray:
        data = np.array(data)
    assert len(data.shape) == 3, 'data must be in 3-D'
    assert type(singular_values) == int
    z, y, x = data.shape
    svd_data = data.reshape((z,y*x))
    U, s, V = np.linalg.svd(svd_data,full_matrices=full_matrices)
    s_approx = s.copy()
    s_approx[singular_values:] = 0

    filtered_image = np.dot(U, np.dot(np.diag(s_approx),V))
    filtered_image = filtered_image.reshape((z, y, x))

    if full_result:
        return filtered_image, (U, s, V)
    else:
        return filtered_image

def calc_svd(usv,shape=(3,100,100),value=0,singular_values=10):
    U, s, V = usv
    if value == -1:
        s_approx = s.copy()
        s_approx[singular_values:] = 0
    else:
        s_approx = np.zeros(s.shape)
        s_approx[value] = s[value]
    filtered_image = np.dot(U, np.dot(np.diag(s_approx),V))
    filtered_image = filtered_image.reshape(shape)
    return filtered_image

class Image(object):
    # def __getattribute__(self, item):
    #     if item == 'image':
    #         if self.svd_image is not None:
    #             return self.svd_image
    #         else:
    #             return self.raw_image
    #     else:
    #         object.__getattribute__(self, item)

    def __init__(self, data):

        assert isinstance(data, np.ndarray)

        self.raw_image = data
        self.shape = data.shape
        self.dtype = data.dtype
        self.svd_image = None
        self.svd_image_single = None
        self.usv = None
        self.singular_values = None

    def calc_svd(self, singular_values=None, anscombe_results=True):
        if singular_values is None:
            if self.singular_values is None:
                singular_values = 10
            else:
                singular_values = self.singular_values

        pool = ThreadPool(processes=1)
        svd_image = self.raw_image

        if anscombe_results:
            svd_image = anscombe(svd_image)

        args = (svd_image, False, singular_values, True)
        threaded_svd_calculation = pool.apply_async(svd_filter, args)
        svd_image, self.usv = threaded_svd_calculation.get()

        if anscombe_results:
            svd_image = inverse_anscombe(svd_image)

        self.svd_image = svd_image

        return self.svd_image

    def calc_svd_single(self,singular_value=-1):
        if self.usv is None:
            self.calc_svd()
        pool = ThreadPool(processes=1)
        args = (self.usv, self.shape, singular_value, self.singular_values)
        svd_image_single = pool.apply_async(calc_svd, args)
        self.svd_image_single = svd_image_single.get()
        min_val = self.svd_image_single.min()
        if min_val < 0:
            self.svd_image_single -= min_val
        return self.svd_image_single