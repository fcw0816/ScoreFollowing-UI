import cv2 as cv
import json
import matplotlib.cm as cm
import numpy as np
import os 
import random
import soundfile as sf
import torch
import tempfile
import time

from torch import nn
from natsort import natsorted

# Filter, Filterbank, LogarithmicFilterbank, TriangularFilter until "###########" are adapted from the GitHub repository:
# https://github.com/CPJKU/madmom
FILTER_DTYPE = np.float32
A4 = 440.

class Filter(np.ndarray):
    """
    Generic Filter class.

    Parameters
    ----------
    data : 1D numpy array
        Filter data.
    start : int, optional
        Start position (see notes).
    norm : bool, optional
        Normalize the filter area to 1.

    Notes
    -----
    The start position is mandatory if a Filter should be used for the creation
    of a Filterbank.

    """
    # pylint: disable=super-on-old-class
    # pylint: disable=super-init-not-called
    # pylint: disable=attribute-defined-outside-init

    def __init__(self, data, start=0, norm=False):
        # this method is for documentation purposes only
        pass

    def __new__(cls, data, start=0, norm=False):
        # input is an numpy ndarray instance
        if isinstance(data, np.ndarray):
            # cast as Filter
            obj = np.asarray(data, dtype=FILTER_DTYPE).view(cls)
        else:
            raise TypeError('wrong input data for Filter, must be np.ndarray')
        # right now, allow only 1D
        if obj.ndim != 1:
            raise NotImplementedError('please add multi-dimension support')
        # normalize
        if norm:
            obj /= np.sum(obj)
        # set attributes
        obj.start = int(start)
        obj.stop = int(start + len(data))
        # return the object
        return obj

    @classmethod
    def band_bins(cls, bins, **kwargs):
        """
        Must yield the center/crossover bins needed for filter creation.

        Parameters
        ----------
        bins : numpy array
            Center/crossover bins used for the creation of filters.
        kwargs : dict, optional
            Additional parameters for for the creation of filters
            (e.g. if the filters should overlap or not).

        """
        raise NotImplementedError('needs to be implemented by sub-classes')

    @classmethod
    def filters(cls, bins, norm, **kwargs):
        """
        Create a list with filters for the given bins.

        Parameters
        ----------
        bins : list or numpy array
            Center/crossover bins of the filters.
        norm : bool
            Normalize the area of the filter(s) to 1.
        kwargs : dict, optional
            Additional parameters passed to :func:`band_bins`
            (e.g. if the filters should overlap or not).

        Returns
        -------
        filters : list
            Filter(s) for the given bins.

        """
        # generate a list of filters for the given center/crossover bins
        filters = []
        for filter_args in cls.band_bins(bins, **kwargs):
            # create a filter and append it to the list
            filters.append(cls(*filter_args, norm=norm))
        # return the filters
        return filters
    
class Filterbank(np.ndarray):
    """
    Generic filterbank class.

    A Filterbank is a simple numpy array enhanced with several additional
    attributes, e.g. number of bands.

    A Filterbank has a shape of (num_bins, num_bands) and can be used to
    filter a spectrogram of shape (num_frames, num_bins) to (num_frames,
    num_bands).

    Parameters
    ----------
    data : numpy array, shape (num_bins, num_bands)
        Data of the filterbank .
    bin_frequencies : numpy array, shape (num_bins, )
        Frequencies of the bins [Hz].

    Notes
    -----
    The length of `bin_frequencies` must be equal to the first dimension
    of the given `data` array.

    """
    # pylint: disable=super-on-old-class
    # pylint: disable=super-init-not-called
    # pylint: disable=attribute-defined-outside-init

    def __init__(self, data, bin_frequencies):
        # this method is for documentation purposes only
        pass

    def __new__(cls, data, bin_frequencies):
        # input is an numpy ndarray instance
        if isinstance(data, np.ndarray) and data.ndim == 2:
            # cast as Filterbank
            obj = np.asarray(data, dtype=FILTER_DTYPE).view(cls)
        else:
            raise TypeError('wrong input data for Filterbank, must be a 2D '
                            'np.ndarray')
        # set bin frequencies
        if len(bin_frequencies) != obj.shape[0]:
            raise ValueError('`bin_frequencies` must have the same length as '
                             'the first dimension of `data`.')
        obj.bin_frequencies = np.asarray(bin_frequencies, dtype=np.float)
        # return the object
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        # set default values here
        self.bin_frequencies = getattr(obj, 'bin_frequencies', None)

    @classmethod
    def _put_filter(cls, filt, band):
        """
        Puts a filter in the band, internal helper function.

        Parameters
        ----------
        filt : :class:`Filter` instance
            Filter to be put into the band.
        band : numpy array
            Band in which the filter should be put.

        Notes
        -----
        The `band` must be an existing numpy array where the filter `filt` is
        put in, given the position of the filter. Out of range filters are
        truncated. If there are non-zero values in the filter band at the
        respective positions, the maximum value of the `band` and the filter
        `filt` is used.

        """
        if not isinstance(filt, Filter):
            raise ValueError('unable to determine start position of Filter')
        # determine start and stop positions
        start = filt.start
        stop = start + len(filt)
        # truncate the filter if it starts before the 0th band bin
        if start < 0:
            filt = filt[-start:]
            start = 0
        # truncate the filter if it ends after the last band bin
        if stop > len(band):
            filt = filt[:-(stop - len(band))]
            stop = len(band)
        # put the filter in place
        filter_position = band[start:stop]
        # TODO: if needed, allow other handling (like summing values)
        np.maximum(filt, filter_position, out=filter_position)

    @classmethod
    def from_filters(cls, filters, bin_frequencies):
        """
        Create a filterbank with possibly multiple filters per band.

        Parameters
        ----------
        filters : list (of lists) of Filters
            List of Filters (per band); if multiple filters per band are
            desired, they should be also contained in a list, resulting in a
            list of lists of Filters.
        bin_frequencies : numpy array
            Frequencies of the bins (needed to determine the expected size of
            the filterbank).

        Returns
        -------
        filterbank : :class:`Filterbank` instance
            Filterbank with respective filter elements.

        """
        # create filterbank
        fb = np.zeros((len(bin_frequencies), len(filters)))
        # iterate over all filters
        for band_id, band_filter in enumerate(filters):
            # get the band's corresponding slice of the filterbank
            band = fb[:, band_id]
            # if there's a list of filters for the current band, put them all
            # into this band
            if isinstance(band_filter, list):
                for filt in band_filter:
                    cls._put_filter(filt, band)
            # otherwise put this filter into that band
            else:
                cls._put_filter(band_filter, band)
        # create Filterbank and cast as class where this method was called from
        return Filterbank.__new__(cls, fb, bin_frequencies)

    @property
    def num_bins(self):
        """Number of bins."""
        return self.shape[0]

    @property
    def num_bands(self):
        """Number of bands."""
        return self.shape[1]

    @property
    def corner_frequencies(self):
        """Corner frequencies of the filter bands."""
        freqs = []
        for band in range(self.num_bands):
            # get the non-zero bins per band
            bins = np.nonzero(self[:, band])[0]
            # append the lowest and highest bin
            freqs.append([np.min(bins), np.max(bins)])
        # map to frequencies
        return bins2frequencies(freqs, self.bin_frequencies)

    @property
    def center_frequencies(self):
        """Center frequencies of the filter bands."""
        freqs = []
        for band in range(self.num_bands):
            # get the non-zero bins per band
            bins = np.nonzero(self[:, band])[0]
            min_bin = np.min(bins)
            max_bin = np.max(bins)
            # if we have a uniform filter, use the center bin
            if self[min_bin, band] == self[max_bin, band]:
                center = int(min_bin + (max_bin - min_bin) / 2.)
            # if we have a filter with a peak, use the peak bin
            else:
                center = min_bin + np.argmax(self[min_bin: max_bin, band])
            freqs.append(center)
        # map to frequencies
        return bins2frequencies(freqs, self.bin_frequencies)

    @property
    def fmin(self):
        """Minimum frequency of the filterbank."""
        return self.bin_frequencies[np.nonzero(self)[0][0]]

    @property
    def fmax(self):
        """Maximum frequency of the filterbank."""
        return self.bin_frequencies[np.nonzero(self)[0][-1]]

FMIN = 30.
FMAX = 17000.
NUM_BANDS = 12
NORM_FILTERS = True
UNIQUE_FILTERS = True

class LogarithmicFilterbank(Filterbank):
    """
    Logarithmic filterbank class.

    Parameters
    ----------
    bin_frequencies : numpy array
        Frequencies of the bins [Hz].
    num_bands : int, optional
        Number of filter bands (per octave).
    fmin : float, optional
        Minimum frequency of the filterbank [Hz].
    fmax : float, optional
        Maximum frequency of the filterbank [Hz].
    fref : float, optional
        Tuning frequency of the filterbank [Hz].
    norm_filters : bool, optional
        Normalize the filters to area 1.
    unique_filters : bool, optional
        Keep only unique filters, i.e. remove duplicate filters resulting
        from insufficient resolution at low frequencies.
    bands_per_octave : bool, optional
        Indicates whether `num_bands` is given as number of bands per octave
        ('True', default) or as an absolute number of bands ('False').

    Notes
    -----
    `num_bands` sets either the number of bands per octave or the total number
    of bands, depending on the setting of `bands_per_octave`. `num_bands` is
    used to set also the number of bands per octave to keep the argument for
    all classes the same. If 12 bands per octave are used, a filterbank with
    semitone spacing is created.

    """
    # pylint: disable=super-on-old-class
    # pylint: disable=super-init-not-called
    # pylint: disable=attribute-defined-outside-init

    NUM_BANDS_PER_OCTAVE = 12

    def __init__(self, bin_frequencies, num_bands=NUM_BANDS_PER_OCTAVE,
                 fmin=FMIN, fmax=FMAX, fref=A4, norm_filters=NORM_FILTERS,
                 unique_filters=UNIQUE_FILTERS, bands_per_octave=True):
        # this method is for documentation purposes only
        pass

    def __new__(cls, bin_frequencies, num_bands=NUM_BANDS_PER_OCTAVE,
                fmin=FMIN, fmax=FMAX, fref=A4, norm_filters=NORM_FILTERS,
                unique_filters=UNIQUE_FILTERS, bands_per_octave=True):
        # pylint: disable=arguments-differ
        # decide whether num_bands is bands per octave or total number of bands
        if bands_per_octave:
            num_bands_per_octave = num_bands
            # get a list of frequencies with logarithmic scaling
            frequencies = log_frequencies(num_bands, fmin, fmax, fref)
            # convert to bins
            bins = frequencies2bins(frequencies, bin_frequencies,
                                    unique_bins=unique_filters)
        else:
            # iteratively get the number of bands
            raise NotImplementedError("please implement `num_bands` with "
                                      "`bands_per_octave` set to 'False' for "
                                      "LogarithmicFilterbank")
        # get overlapping triangular filters
        filters = TriangularFilter.filters(bins, norm=norm_filters,
                                           overlap=True)
        # create a LogarithmicFilterbank from the filters
        obj = cls.from_filters(filters, bin_frequencies)
        print(obj.shape)
        # set additional attributes
        obj.fref = fref
        obj.num_bands_per_octave = num_bands_per_octave
        # return the object
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        # set default values here
        self.num_bands_per_octave = getattr(obj, 'num_bands_per_octave',
                                            self.NUM_BANDS_PER_OCTAVE)
        self.fref = getattr(obj, 'fref', A4)
  
class TriangularFilter(Filter):
    """
    Triangular filter class.

    Create a triangular shaped filter with length `stop`, height 1 (unless
    normalized) with indices <= `start` set to 0.

    Parameters
    ----------
    start : int
        Start bin of the filter.
    center : int
        Center bin of the filter.
    stop : int
        Stop bin of the filter.
    norm : bool, optional
        Normalize the area of the filter to 1.

    """
    # pylint: disable=super-on-old-class
    # pylint: disable=super-init-not-called
    # pylint: disable=attribute-defined-outside-init

    def __init__(self, start, center, stop, norm=False):
        # this method is for documentation purposes only
        pass

    def __new__(cls, start, center, stop, norm=False):
        # pylint: disable=arguments-differ
        # center must be between start & stop
        if not start <= center < stop:
            raise ValueError('`center` must be between `start` and `stop`')
        # cast variables to int
        center = int(center)
        start = int(start)
        stop = int(stop)
        # make center and stop relative
        center -= start
        stop -= start
        # create filter
        data = np.zeros(stop)
        # rising edge (without the center)
        data[:center] = np.linspace(0, 1, center, endpoint=False)
        # falling edge (including the center, but without the last bin)
        data[center:] = np.linspace(1, 0, stop - center, endpoint=False)
        # cast to TriangularFilter
        obj = Filter.__new__(cls, data, start, norm)
        # set the center bin
        obj.center = start + center
        # return the filter
        return obj

    @classmethod
    def band_bins(cls, bins, overlap=True):
        """
        Yields start, center and stop bins for creation of triangular filters.

        Parameters
        ----------
        bins : list or numpy array
            Center bins of filters.
        overlap : bool, optional
            Filters should overlap (see notes).

        Yields
        ------
        start : int
            Start bin of the filter.
        center : int
            Center bin of the filter.
        stop : int
            Stop bin of the filter.

        Notes
        -----
        If `overlap` is 'False', the `start` and `stop` bins of the filters
        are interpolated between the centre bins, normal rounding applies.

        """
        # pylint: disable=arguments-differ
        # make sure enough bins are given
        if len(bins) < 3:
            raise ValueError('not enough bins to create a TriangularFilter')
        # yield the bins
        index = 0
        while index + 3 <= len(bins):
            # get start, center and stop bins
            start, center, stop = bins[index: index + 3]
            # create non-overlapping filters
            if not overlap:
                # re-arrange the start and stop positions
                start = int(np.floor((center + start) / 2.))
                stop = int(np.ceil((center + stop) / 2.))
            # consistently handle too-small filters
            if stop - start < 2:
                center = start
                stop = start + 1
            # yield the bins and continue
            yield start, center, stop
            # increase counter
            index += 1

def frequencies2bins(frequencies, bin_frequencies, unique_bins=False):
    """
    Map frequencies to the closest corresponding bins.

    Parameters
    ----------
    frequencies : numpy array
        Input frequencies [Hz].
    bin_frequencies : numpy array
        Frequencies of the (FFT) bins [Hz].
    unique_bins : bool, optional
        Return only unique bins, i.e. remove all duplicate bins resulting from
        insufficient resolution at low frequencies.

    Returns
    -------
    bins : numpy array
        Corresponding (unique) bins.

    Notes
    -----
    It can be important to return only unique bins, otherwise the lower
    frequency bins can be given too much weight if all bins are simply summed
    up (as in the spectral flux onset detection).

    """
    # cast as numpy arrays
    frequencies = np.asarray(frequencies)
    bin_frequencies = np.asarray(bin_frequencies)
    # map the frequencies to the closest bins
    # solution found at: http://stackoverflow.com/questions/8914491/
    indices = bin_frequencies.searchsorted(frequencies)
    indices = np.clip(indices, 1, len(bin_frequencies) - 1)
    left = bin_frequencies[indices - 1]
    right = bin_frequencies[indices]
    indices -= frequencies - left < right - frequencies
    # only keep unique bins if requested
    if unique_bins:
        indices = np.unique(indices)
    # return the (unique) bin indices of the closest matches
    # print(indices, len(indices))
    return indices
        
def bins2frequencies(bins, bin_frequencies):
    # map the frequencies to spectrogram bins
    return np.asarray(bin_frequencies, dtype=np.float)[np.asarray(bins)]

def fft_frequencies(num_fft_bins, sample_rate):
    return np.fft.fftfreq(num_fft_bins * 2, 1. / sample_rate)[:num_fft_bins]

def log_frequencies(bands_per_octave, fmin, fmax, fref=A4):
    """
    Returns frequencies aligned on a logarithmic frequency scale.

    Parameters
    ----------
    bands_per_octave : int
        Number of filter bands per octave.
    fmin : float
        Minimum frequency [Hz].
    fmax : float
        Maximum frequency [Hz].
    fref : float, optional
        Tuning frequency [Hz].

    Returns
    -------
    log_frequencies : numpy array
        Logarithmically spaced frequencies [Hz].

    Notes
    -----
    If `bands_per_octave` = 12 and `fref` = 440 are used, the frequencies are
    equivalent to MIDI notes.

    """
    # get the range
    left = np.floor(np.log2(float(fmin) / fref) * bands_per_octave)
    right = np.ceil(np.log2(float(fmax) / fref) * bands_per_octave)
    # print(left, right)
    # generate frequencies
    frequencies = fref * 2. ** (np.arange(left, right) /
                                float(bands_per_octave))
    # print(frequencies, len(frequencies))
    # filter frequencies
    # needed, because range might be bigger because of the use of floor/ceil
    frequencies = frequencies[np.searchsorted(frequencies, fmin):]
    frequencies = frequencies[:np.searchsorted(frequencies, fmax, 'right')]
    # print(frequencies, len(frequencies))
    # return
    return frequencies
################################################

SCORE_WIDTH = 835
SCORE_HEIGHT = 1181
PAGE_TURNING_THRESHOLD = 0.8
COOLDOWN = 20

SAMPLE_RATE = 22050
FRAME_SIZE = 2048
HOP_SIZE = 1102
FPS = SAMPLE_RATE/HOP_SIZE

CLASS_MAPPING = {0: 'Note', 1: 'Bar', 2: 'System'}
COLORS = [(0, 0, 1), (0, 1, 0), (1, 0, 0)]

def find_system_edge(curr_y, edge_list):
    diff_array = np.abs(edge_list - curr_y)
    
    min_diff_idx = np.argmin(diff_array)
    # print(min_diff_idx)
    return  min_diff_idx

def load_piece_for_inference(scale_width, mode, path, piece_name, crop_path):
    if mode == "fullpage": 
        padded_scores, org_scores, crop_padded_scores, crop_org_scores, pad = load_piece(path, piece_name, crop_path)
    else:
        padded_scores, org_scores, crop_padded_scores, crop_org_scores, pad = load_bipiece(path, piece_name, crop_path)
        
    scores = 1 - np.array(padded_scores, dtype=np.float32) / 255.

    # scale scores
    scaled_score = []
    # print(scores[0].shape)
        
    for score in scores:
        scaled_score.append(cv.resize(score, (scale_width, scale_width), interpolation=cv.INTER_AREA))

    score = np.stack(scaled_score)

    org_scores_rgb = []
    for org_score in org_scores:
        org_score = cv.cvtColor(np.array(org_score, dtype=np.float32) / 255. , cv.COLOR_GRAY2BGR)

        org_scores_rgb.append(org_score)
    
    crop_scores = 1 - np.array(crop_padded_scores, dtype=np.float32) / 255.

    # scale scores
    crop_scaled_score = []
    
    for crop_score in crop_scores:
        crop_scaled_score.append(cv.resize(crop_score, (scale_width, scale_width), interpolation=cv.INTER_AREA))

    crop_score = np.stack(crop_scaled_score)

    
    crop_org_scores_rgb = []

    for crop_org_score in crop_org_scores:
        crop_org_score = cv.cvtColor(np.array(crop_org_score, dtype=np.float32) / 255., cv.COLOR_GRAY2BGR)

        crop_org_scores_rgb.append(crop_org_score)
        
    if mode == "fullpage": 
        scale_factor = scores[0].shape[0] / scale_width
    else:
        scale_factor = crop_scores[0].shape[1] / scale_width

    return org_scores_rgb, score, crop_org_scores_rgb, crop_score, pad, scale_factor
    
def load_piece(path, piece_name, crop_path):
    images = []
    images_path = []
    
    image_path = os.path.join(path, piece_name)
    l = os.listdir(image_path)
    l = natsorted(l)
    for i in l:

        path = os.path.join(image_path, i)
        image = cv.imread(path)
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

        images.append(image)
        images_path.append(path)
    images = np.array(images)
    
    if len(images.shape) == 2:
        h, w = images.shape
        
        image = np.array([image])
    else:
        # print(images.shape)
        n, h, w = images.shape
    # print(h, w)
    dim_diff = np.abs(h - w)
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2

    pad = ((0, 0), (0, 0), (pad1, pad2))
    # Add padding
    padded_scores = np.pad(images, pad, mode="constant", constant_values=255)

    crop_images = []
    cropimage_path = os.path.join(crop_path, piece_name)
    l = os.listdir(cropimage_path)
    l = natsorted(l)
    for i in l:
        # print(i)
        if i.endswith(".jpg"):
            path = os.path.join(cropimage_path, i)
            image = cv.imread(path)
            image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
            crop_images.append(image)
    crop_images = np.array(crop_images)
    
    if len(crop_images.shape) == 2:
        h, w = crop_images.shape
        image = np.array([image])
    else:
        # print(crop_images.shape)
        n, h, w = crop_images.shape
        
    dim_diff = np.abs(h - w)
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2

    # Determine padding
    pad = ((0, 0), (0, 0), (pad1, pad2))
    # print(image.shape)
    # Add padding
    crop_padded_scores = np.pad(crop_images, pad, mode="constant", constant_values=255)

    return padded_scores, images, crop_padded_scores, crop_images, pad1
    
def load_bipiece(path, piece_name, crop_path):
        
    images = []
    images_path = []
    
    image_path = os.path.join(path, piece_name)
    l = os.listdir(image_path)
    l = natsorted(l)
    for i in l:
        # print(i)
        path = os.path.join(image_path, i)
        image = cv.imread(path)
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        # cv.imshow("ORINGIN", image)
        # cv.waitKey(0)
        images.append(image)
        images_path.append(path)
        
    images = np.array(images)

    if len(images.shape) == 2:
        h, w = images.shape
        images = np.array([images])
    else:
        # print(images.shape)
        n, h, w = images.shape
    
    pad = ((0, 0), (0, 0), (0, 0))
    # Add padding
    padded_scores = np.pad(images, pad, mode="constant", constant_values=255)

    crop_images = []
    cropimage_path = os.path.join(crop_path, piece_name)
    
    l = os.listdir(cropimage_path)
    l = natsorted(l)
    for i in l:
        # print(i)
        if i.endswith(".jpg"):
            path = os.path.join(cropimage_path, i)
            image = cv.imread(path)
            image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
            # cv.imshow("CROPIMG", image)
            # cv.waitKey(0)
            crop_images.append(image)
    crop_images = np.array(crop_images)
    
    
        
    bipage = []
    null_page = np.ones_like(crop_images[0, :, :]) * 255
    
    # scores = np.concatenate((crop_images, null_page), axis=0)
    # print(scores.shape, null_page.shape)
    for i in range(len(crop_images)):
        # print(i, np.concatenate((scores[i], scores[i+1]), axis=-1).shape)
        bipage.append(np.concatenate((crop_images[i], null_page), axis=-1))  
        print(len(bipage))
        # cv.imshow("BIPAGE", bipage[i])
        # cv.waitKey(0)
    bipage = np.array(bipage)
    print(bipage.shape)
        
    if len(bipage.shape) == 2:
        h, w = bipage.shape
        bipage = np.array([bipage])
    else:
        # print(bipage.shape)
        n, h, w = bipage.shape
        
    dim_diff = np.abs(h - w)
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2

    # Determine padding
    pad = ((0, 0), (pad1, pad2), (0, 0))
    # print(image.shape)
    # Add padding
    crop_padded_scores = np.pad(bipage, pad, mode="constant", constant_values=255)
    
    # wav_path = os.path.join(audio_path, piece_name + '.wav')
    # signal = load_wav(wav_path, sr=SAMPLE_RATE)
    
    return padded_scores, images, crop_padded_scores, bipage, pad1

def load_pretrained_model(param_path, mode):
    from utils.cyolo import Model
            
    param_dir = os.path.dirname(param_path)

    with open(os.path.join(param_dir, 'net_config.json'), 'r') as f:
        config = json.load(f)

    network = Model(config)
    try:
        network.load_state_dict(torch.load(param_path, map_location=lambda storage, location: storage))
    except:
        network = nn.parallel.DataParallel(network)
        network.load_state_dict(torch.load(param_path, map_location=lambda storage, location: storage))
        network = network.module

    return network
    
def xywh2xyxy(x):
    """taken from https://github.com/ultralytics/yolov5/blob/master/utils/general.py"""
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y
    
def box_iou(box1, box2):
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (ndarray[N, 4]): First set of boxes.
        box2 (ndarray[M, 4]): Second set of boxes.
    Returns:
        iou (ndarray[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """
    
    def xywh2xyxy_(box):
        x, y, w, h = box[:, 0], box[:, 1], box[:, 2], box[:, 3]
        x1 = x - w / 2
        y1 = y - h / 2
        x2 = x + w / 2
        y2 = y + h / 2
        return np.stack((x1, y1, x2, y2), axis=-1)

    def box_area(box):
        # box is expected to be in (x1, y1, x2, y2) format
        return (box[2] - box[0]) * (box[3] - box[1])

    box1 = xywh2xyxy_(box1)
    box2 = xywh2xyxy_(box2)
    # Compute area of both sets of boxes
    area1 = box_area(box1.T)  # box1 is shape (N, 4)
    area2 = box_area(box2.T)  # box2 is shape (M, 4)

    # Find the coordinates of the intersection boxes
    inter_top_left = np.maximum(box1[:, None, :2], box2[:, :2])  # (N, M, 2)
    inter_bottom_right = np.minimum(box1[:, None, 2:], box2[:, 2:])  # (N, M, 2)

    # Compute the width and height of the intersection boxes
    inter_wh = np.maximum(inter_bottom_right - inter_top_left, 0)  # (N, M, 2)

    # Compute the area of the intersection boxes
    inter_area = inter_wh[:, :, 0] * inter_wh[:, :, 1]  # (N, M)

    # Compute the union area
    union_area = area1[:, None] + area2 - inter_area  # (N, M)

    # Compute the IoU
    iou = inter_area / union_area

    return iou

def plot_box(x, img, color=None, label=None, line_thickness=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv.rectangle(img, c1, c2, color, thickness=tl, lineType=cv.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv.rectangle(img, c1, c2, color, -1, cv.LINE_AA)  # filled
        cv.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv.LINE_AA)

def prepare_spec_for_render(spec, score, scale_factor=2):
    spec_excerpt = cv.resize(np.flipud(spec), (spec.shape[1] * scale_factor, spec.shape[0] * scale_factor))
    perf_img = np.pad(cm.viridis(spec_excerpt)[:, :, :3],
                    ((score.shape[0] // 2 - spec_excerpt.shape[0] // 2 + 1,
                        score.shape[0] // 2 - spec_excerpt.shape[0] // 2),
                    (20, 20), (0, 0)), mode="constant")
    # print(cm.viridis(spec_excerpt)[:, :, :3].shape, (score.shape[0] // 2 - spec_excerpt.shape[0] // 2 + 1))

    return perf_img

def write_video(images, fn_output='output.mp4', frame_rate=20, overwrite=False, progress=None):
    """Takes a list of images and interprets them as frames for a video.

    Source: http://tsaith.github.io/combine-images-into-a-video-with-python-3-and-opencv-3.html
    """
    height, width, _ = images[0].shape

    if overwrite:
        if os.path.exists(fn_output):
            os.remove(fn_output)

    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    out = cv.VideoWriter(fn_output, fourcc, frame_rate, (width, height))

    for i, cur_image in enumerate(images):
        if progress != None:
            progress.setValue(int(100 * ( (i+1)/len(images))))
        frame = cv.resize(cur_image, (width, height))
        out.write(frame)  # Write out frame to video

    # Release everything if job is finished
    out.release()

    return fn_output

def create_video(observation_images, signal, piece_name, fps, sample_rate, path="../videos", tag="", progress=None):
    if progress != None:
        progress.setValue(0)
    
    # create temp wavfile
    wav_path = os.path.join(tempfile.gettempdir(), str(time.time()) + '.wav')
    sf.write(wav_path, signal, samplerate=sample_rate)

    path_video = write_video(observation_images,
                             fn_output=os.path.join(tempfile.gettempdir(), str(time.time()) + '.mp4'),
                             frame_rate=fps, overwrite=True, progress=progress)

    # mux video and audio with ffmpeg
    mux_video_audio(path_video, wav_path, path_output=os.path.join(path, f'{piece_name}{tag}.mp4'))
    if progress != None:
        progress.setValue(100)
    # clean up
    os.remove(path_video)
    
def mux_video_audio(path_video, path_audio, path_output='output_audio.mp4'):
    """Use FFMPEG to mux video with audio recording."""
    from subprocess import check_call

    # check_call(["ffmpeg", "-y", "-i", path_video, "-i", path_audio, "-shortest", "-c:v", "h264", path_output])
    check_call([r"D:\scorefollowersystem\cyolo_score_following\ffmpeg\bin\ffmpeg.exe", "-y", "-i", path_video, "-i", path_audio, "-shortest", "-c:v", "h264", path_output])

def motif_iou(motif, position, size):

        h, w, n = size
        motif = np.array(motif)
        position = np.array(position)
       
        position = np.tile(position, (motif.shape[0], 1)).astype(float)

        if motif.shape == position.shape:
            label = motif[:, 0]
            motif = motif[:, 1:].astype(float)
            # print(label, motif, position)
            motif[:, ::2] *= w
            motif[:, 1::2] *= h

            metrics = box_iou(motif, position)
            
            if np.sum(metrics) > 0:
                return True, label[np.argmax(metrics)]
            else: 
                return False, "None"

        else:
            return False, "None"
    

    
   