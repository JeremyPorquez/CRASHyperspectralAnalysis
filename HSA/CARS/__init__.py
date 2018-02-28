import numpy as np

import pandas as pd
from .crikit.utils import als_methods as als
from .crikit import pre as crikit
import scipy.interpolate as spi
import scipy.signal
import scipy
import matplotlib.pyplot as plt
import multiprocessing
from multiprocessing.pool import ThreadPool


def getCARSPhase(data, dataNRB=''):
    '''Old version'''

    yR = data ** 0.5
    yNR = dataNRB ** 0.5
    yR = pd.Series(yR)
    yNR = pd.Series(yNR)

    sf = 4  # stretch factor
    xyR = np.zeros(sf * len(yR))
    xyNR = np.zeros(sf * len(yR))
    xyR = pd.Series(xyR)
    xyNR = pd.Series(xyNR)

    a = len(xyR) / 4
    b = 3 * len(xyR) / 4
    a = (sf - 1) * (len(yR) / 2)
    b = (sf + 1) * (len(yR) / 2)

    if (len(yR) % 2 != 0):
        b += 1

    xyR.iloc[:a] = yR.values[0]
    xyR.iloc[a:b] = yR.values
    xyR.iloc[b:] = yR.values[-1]
    xyNR.iloc[:a] = yNR.values[0]
    xyNR.iloc[a:b] = yNR.values
    xyNR.iloc[b:] = yNR.values[-1]

    logxyR = np.log(xyR.values)
    logxyNR = np.log(xyNR.values)

    xyRfft = np.fft.fftshift(np.fft.fft(logxyR))
    xyNRfft = np.fft.fftshift(np.fft.fft(logxyNR))
    xyRfft = pd.Series(xyRfft)
    xyNRfft = pd.Series(xyNRfft)

    # t<0 NR, t>=0 Resonant
    fft = pd.Series(xyRfft)
    fft.iloc[:len(fft) / 2] = xyNRfft[:len(fft) / 2]
    #     fft.iloc[len(fft)/2:] += xyNRfft[len(fft)/2:]
    fft = np.fft.fftshift(fft)
    eta = fft

    ifft = np.fft.ifft(fft)

    phase = 2 * np.imag(ifft - logxyR / 2)
    raman = xyR * np.sin(phase)

    # trim data to original
    raman = raman[a:b]

    return raman  # ,phase,eta

def getCorrectedCARSPhase(I_CARS,
                          I_REF='',
                          PHASE_OFFSET=0,
                          NORM_BY_NRB=1,
                          SMOOTHNESS_PARAM=1e5,
                          ASYM_PARAM=1e-4,
                          SAVGOL_WINDOW=601
                         ):
    Retrieved_complex_spectrum_w_reference = crikit.kkrelation(I_REF,I_CARS,PHASE_OFFSET,NORM_BY_NRB) # Complex spectrum
    Error_phase = als.als_baseline(np.angle(Retrieved_complex_spectrum_w_reference), SMOOTHNESS_PARAM,ASYM_PARAM)[0]
    Phase_Corrected = Retrieved_complex_spectrum_w_reference*(1/np.exp(crikit.hilbertfft(Error_phase).imag) * \
                                                          np.exp(-1j*Error_phase))
    Corrected = 1/(scipy.signal.savgol_filter(np.real(Phase_Corrected),SAVGOL_WINDOW,2,axis=0))*Phase_Corrected
    return Corrected.imag

def getCorrectedCARSPhaseImage(img,
                               I_REF='',
                               PHASE_OFFSET=0,
                               NORM_BY_NRB=1,
                               SMOOTHNESS_PARAM=1e5,
                               ASYM_PARAM=1e-4,
                               SAVGOL_WINDOW=601
                               ):
    assert isinstance(img,np.ndarray)

    z,y,x = img.shape

    num_processors = multiprocessing.cpu_count()
    pool = ThreadPool()
    split_data = np.array_split(img.reshape(z,y*x), num_processors)

    results = []
    for data in split_data:

        def proc(d):
            z, pixels = d.shape
            for pix in range(pixels):
                I_CARS = d[:,pix]
                d[pix] = getCorrectedCARSPhase(I_CARS,
                                               I_REF,
                                               PHASE_OFFSET=0,
                                               NORM_BY_NRB=1,
                                               SMOOTHNESS_PARAM=1e5,
                                               ASYM_PARAM=1e-4,
                                               SAVGOL_WINDOW=601)
                print(pix)

        results.append(pool.apply_async(proc, (data)))

    for result in results:
        result.get()   #makes sure results are in before numpy restacks the data

    retrieved_img = np.vstack(split_data)

    return retrieved_img

def getCorrectedCARSPhaseXY(CARS,
                            REF='',
                            PHASE_OFFSET=0,
                            NORM_BY_NRB=1,
                            SMOOTHNESS_PARAM=1e5,
                            ASYM_PARAM=1e-4,
                            SAVGOL_WINDOW=601,
                            kind="cubic"
                            ):
    xCARS, yCARS = CARS
    xRef, yRef = REF
    CARS_function = spi.interp1d(xCARS, yCARS, fill_value="extrapolate")
    Ref_function = spi.interp1d(xRef, yRef, fill_value="extrapolate")
    if SAVGOL_WINDOW > len(xCARS):
        if len(xCARS) % 2 == 0:
            SAVGOL_WINDOW = len(xCARS) - 1
        else :
            SAVGOL_WINDOW = len(xCARS)
    phase = getCorrectedCARSPhase(I_CARS=CARS_function(xCARS),I_REF=Ref_function(xCARS),PHASE_OFFSET=PHASE_OFFSET,
                                  NORM_BY_NRB=NORM_BY_NRB,SMOOTHNESS_PARAM=SMOOTHNESS_PARAM,
                                  ASYM_PARAM=ASYM_PARAM,SAVGOL_WINDOW=SAVGOL_WINDOW)
    return phase


def matchData(data, refData):
    # xrefdata must be inside xdata

    try:
        pdData = pd.DataFrame()
        for i in data.columns:
            try:
                xrefData = refData.index
            except:
                xrefData = refData.ix[:, 0]
            y = spi.interp1d(data.index, data[i], fill_value="extrapolate")
            pdData[i] = y(xrefData)
            pdData.index = xrefData

        return pdData

    except:
        xdata = data[0]
        ydata = data[1]
        xrefData = refData[0]
        y = spi.interp1d(xdata, ydata, fill_value="extrapolate")

        return xrefData, y(xrefData)


def getSinglePixelvsZ(image, x, y):
    """Gets all z-values for a given x,y pixel position in a 3d image array.

    Parameters
    ----------
    image : array or numpy array
        Description here...
    x : int
        Position of x-pixel or column number in the array.
        Input from 0 or until the column length of the array.
    y : int
        Position of y-pixel or row number in the array.
        Input from 0 or until the row length of the array.

    Returns
    ----------
    data : ndarray
        The z-values for the given 3d array.

    Examples
    --------
    >>> Z = getSinglePixelvsZ(image,30,50)
    """

    data = []
    for i in range(0, len(image)):
        data.append(image[i][x][y])
    return np.array(data)


def getMaxValues(data, extension='SSM', savefile='temp_2dmaxvalues.csv', save=False):
    if type(data) == str:
        data = extract2ddata(data, extension=extension)

    maxpositions = data.idxmax(axis=0)
    maxvalues = data.max(axis=0)

    maxpositions = np.array(maxpositions)
    maxvalues = np.array(maxvalues)
    if save == 'True':
        maxpositions

    df = pd.DataFrame()
    df['Y'] = data.columns
    df['X'] = maxpositions
    df['value'] = maxvalues

    if save == True:
        df.to_csv(savefile)

    return df

def calibrate(data,min_wavelength=600,max_wavelength=800,wp=12500,stage_min=85,stage_max=90,tolerance=300,calibration_file='calibrationformula.txt'):
    data = data[(data.index > 600) & (data.index < 800)]
    data.index = 1e7 / data.index - wp
    x, y, z = pd.to_numeric(data.columns, errors="coerce"), pd.to_numeric(data.index, errors='coerce'), data.values

    m = getMaxValues(data)
    m['Y'] = pd.to_numeric(m['Y'], errors='coerce')
    m['X'] = pd.to_numeric(m['X'], errors='coerce')
    m = m[m.Y > stage_min]
    m = m[m.Y < stage_max]

    try:
        baseline = scipy.signal.savgol_filter(m['X'], window_length=1 + len(m['X']) / 2, polyorder=2)
    except:
        baseline = scipy.signal.savgol_filter(m['X'], window_length=len(m['X']) / 2, polyorder=2)
    diff = m['X'] - baseline
    m = m[np.abs(diff) < tolerance]

    fit_calibration = np.polyfit(m['Y'], m['X'], 2)
    x2 = np.linspace(x.min(), x.max(), 1000)

    fig = plt.figure()
    ax = fig.gca()
    ax.set_xlabel('Stage delay (mm)')
    ax.set_ylabel('Vibrational frequency (cm$^{-1}$)')
    ax.contourf(x, y, z, levels=np.linspace(z.min().min(), z.max().max(), 100))
    ax.scatter(m['Y'], m['X'], color='black', alpha=.5)
    ax.plot(x2, np.poly1d(fit_calibration)(x2), color='red', alpha=0.3, lw=2)
    ax.set_ylim(y.min(), y.max())
    ax.set_xlim(x.min(), x.max())

    plt.tight_layout()
    plt.show()

    print(str(m['X'].min()) + ' cm-1')
    s = ['+(%.20f)' % i + '*x' * (2 - idx) for idx, i in enumerate(fit_calibration)]
    si = ''
    for i in s:
        si += i

    f = open(calibration_file, "wb")
    f.write(si)
    f.close()
    print(si)