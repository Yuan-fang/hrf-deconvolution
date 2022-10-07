# -*- coding: utf-8 -*-
"""
Spyder Editor

Author: Yuanfang Zhao
2021 at Donders Institute for Brain, Cognition and Behaviour

"""

import numpy as np


class UserDefinedException(Exception):
    """
    Exception defined by user
    """

    def __init__(self, str):
        """
        

        Parameters
        ----------
        str : string
            A string to indicate the exception.

        Returns
        -------
        None.

        """
        Exception.__init__(self)
        self._str = str


class HrfRetrieval(object):
    def __init__(self, tsData, cond_file, TR, ER=None, window=[-4, 24]):
        """
        

        Parameters
        ----------
        tsData : 1-D or 2-D array
            fMRI timecourse. 1-D array or 2-D array. Each row represents a voxel.
            
        cond_file : str
            text file containing all the events [1-column: event timing; 
                                                 2-colume: event number;
                                                 3-colume: event duration;
                                                 4-colume: event label].
            
        window : list
            window size. Defining the peristimulus interval. unit in seconds.
            The default is [-4, 24].
            
        TR : float number
            Repetition time of the actual scanning
            
        ER : float number
            Effective resolution of the peristimulus interval. unit in seconds.
            TR should be even multple of ER. The default ER = TR
            

        Returns
        -------
        None.

        """
        # fmri time course (1D or 2D array)
        self.tc = tsData

        # number of time points
        self.ntps = tsData.shape[-1]

        # extract info from cond_file
        cond = np.genfromtxt(cond_file, dtype=None, encoding='utf-8')
        self.event_timing = cond['f0']
        self.event_number = cond['f1']

        # number of conditions
        self.nconds = len(np.unique(self.event_number))

        # condition labels and codes
        event_labels = cond['f4']
        indexes = np.unique(self.event_number, return_index=True)[1]
        self.c_labels = [event_labels[index] for index in indexes]
        self.c_codes = np.unique(self.event_number)

        # dict list of conditions
        self.cond = []
        for c_name in self.c_labels:
            self.cond.append({'label': c_name})

        # start and end of the peristimulus window
        self.Tstart = abs(window[0])
        self.Tend = abs(window[1])

        # TR
        self.TR = TR

        # ER
        if (ER is None) or (ER == TR):
            self.ER = TR
        elif (ER is not None) and (ER != TR):
            if np.mod(TR, ER):
                trmult = TR/ER
                self.ER = TR/round(trmult)
                print(
                    'TR must be an even multiple of ER: changing ER to {:0.3f}\n'.format(self.ER))
            else:
                self.ER = ER

        # numper of points for a peristimulus interval
        NPosEst = round(self.Tstart/self.ER)
        NNegEst = round(self.Tend/self.ER)
        self.nHEst = NPosEst + NNegEst

        # number of timepoints before event
        self.nPreStim = np.floor(self.Tstart/self.ER)

        # time scale in s
        self.tscale = np.arange(
            (-self.nPreStim+1)*self.ER, (self.nHEst-self.nPreStim+1)*self.ER, self.ER)

    def getFIRmodel(self):
        """
        Build up the FIR model

        Returns
        -------
        self

        """
        self.model = np.zeros((self.ntps, self.nHEst, self.nconds))

        for c in range(self.nconds):

            # finde nearest TR for each onset
            rounded_onsets = np.round(
                self.event_timing[self.event_number == self.c_codes[c]]/self.ER)*self.ER

            # setup FIR model matrix on the actual TR scale
            trE = np.zeros((self.ntps, self.nHEst))

            # actual TR scale (resolution: 0.1 ms)
            trScale = np.fix(np.arange(0, self.ntps*self.TR, self.TR)*1e4)/1e4

            # builidng FIR model
            for trial in range(len(rounded_onsets)):
                for Ntp in range(self.nHEst):
                    trE[trScale == np.round(
                        (rounded_onsets[trial] + (Ntp-self.nPreStim)*self.ER)*1e4)/1e4, Ntp] = 1

            # assign the model to each condition and a matrix
            self.cond[c]['model'] = trE
            self.model[:, :, c] = trE

    def estimateHRF(self, metric='FIR'):
        """
        

        Parameters
        ----------
        metric : string
            'FIR' or 'average'. The default is 'FIR'. If 'FIR', a FIR GLM will 
            be implemented; if average, simple event selective average will be
            implemented.

        Returns
        -------
        None.

        """
        self.metric = metric
        if self.metric == 'FIR':

            # reshpae the model into 2D array with conditions concacted horizontally
            model = self.model.reshape(self.model.shape[0], -1, order='F')

            # demean columns in the model
            mean = model.mean(axis=0)
            demean_model = model.T - mean[:, np.newaxis]
            demean_model = demean_model.T

            # add column of ones to model non-zero mean
            demean_model_plus_intercept = np.ones(
                (demean_model.shape[0], demean_model.shape[1]+1))
            demean_model_plus_intercept[:, :-1] = demean_model

            # compute beta estimate
            b_est = np.linalg.pinv(demean_model_plus_intercept).dot(self.tc.T)

            # assign the beta estimate to each condition
            for c in range(self.nconds):
                range_st = c*self.nHEst
                range_end = range_st + self.nHEst
                self.cond[c]['FIR_hrf'] = b_est[range_st:range_end].T

        elif self.metric == 'average':
            ind = (self.tscale < 0)
            for c in range(self.nconds):
                model = self.model[:, :, c]
                mean_hrf = self.tc.dot(model)/model.sum(axis=0)
                if mean_hrf.ndim == 1:
                   mean_prestim = mean_hrf[ind].mean()
                   self.cond[c]['average_hrf'] = mean_hrf - mean_prestim 
                elif mean_hrf.ndim == 2:
                     mean_prestim = mean_hrf[:, ind].mean(axis=1)
                     self.cond[c]['average_hrf'] = mean_hrf - mean_prestim[:,None]
        else:
            raise UserDefinedException('Metric can only be FIR/average!')