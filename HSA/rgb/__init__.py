from .. import pyqtgraph as pg
import numpy as np

class ContrastImage(object):
    def __init__(self,calculation_mode='sum'):
        assert isinstance(calculation_mode,str)
        self.calculation_mode = calculation_mode.lower()

        self.r = [pg.ImageView() for i in range(0,3)]
        self.g = [pg.ImageView() for i in range(0,3)]
        self.b = [pg.ImageView() for i in range(0,3)]
        self.rgb = pg.ImageView()

        for color in [self.r,self.g,self.b]:
            for img in color:
                img.timeLine.sigPositionChanged.connect(lambda: self._calculate())

    def _calculate(self, mode=None):
        if mode is None:
            mode = self.calculation_mode

        def func(x, y, mode):
            x, y = x.image[x.currentIndex], y.image[y.currentIndex]
            if mode == 'sum':
                result = x + y
                return result
            elif mode == 'difference':
                result = x - y
                result[result < 0] = 0
                return result


        r = func(self.r[0], self.r[1], mode)
        g = func(self.g[0], self.g[1], mode)
        b = func(self.b[0], self.b[1], mode)
        y, x = r.shape
        rgb = np.zeros((1,y,x,3))
        rgb[0, ..., 0] = r
        rgb[0, ..., 1] = g
        rgb[0, ..., 2] = b

        self.r[2].setImage(r)
        self.g[2].setImage(g)
        self.b[2].setImage(b)
        self.rgb.setImage(rgb)


    def calculate_difference(self):
        self._calculate(mode='difference')

    def calculat_sum(self):
        self._calculate(mode='sum')

    def set_calculation_mode(self,mode='sum'):
        self.calculation_mode = mode.lower()
        self._calculate()

    def set_image(self,image = None):
        if image is None:
            return None
        image = np.swapaxes(image,1,2)
        for img in self.r[:2]:
            img.setImage(image.astype(np.float64),autoLevels=True,autoRange=True)
        for img in self.g[:2]:
            img.setImage(image.astype(np.float64),autoLevels=True,autoRange=True)
        for img in self.b[:2]:
            img.setImage(image.astype(np.float64),autoLevels=True,autoRange=True)


