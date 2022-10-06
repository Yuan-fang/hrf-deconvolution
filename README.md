# HRF Deconvolution Tool

### Introduction
fMRI is an indirect measurement of neural events, which are (presumably) linearly convolved with the HRF (Hemodynamic Response Function) to produce BOLD signal.
![alt text](https://mri-q.com/uploads/3/4/5/7/34572113/convolution-of-3-events_orig.gif)
In most scenarios, we just just use a canonical HRF (retrieved from V1) as the basis function linearly convolved with neural events to foward model the BOLD signal. It works largely well.
However, in some specific scenarios, such as ROI analysis for an event-related design, you may want to get the hemodynamic response for each condition, a process known as deconvolution.
This Python-based tool provides an easy-to-use interface for you to do the deconvolution. 

### Example
Suppose we are interested in retrieving FFA's response to faces, scenes, and chairs. Before we estimate the HRF, we need to firstly prepare the information below:

**tsData**: a 1-D or 2-D numpy array of ROI timecourse data, each row represents a voxel. For 1-D array it could be the averaged signal in the ROI.

**Ev_file**: events timing file containing all the events you are interested in. [1-col: event onset (s); 2-col: event code; 3-col: event duration (s); 4-col:event label] 

| onset | code | duration | label |
|-------|------|----------|-------|
| 2     | 1    | 0.5      | scene |
| 11    | 0    | 0.5      | face  |
| 32    | 0    | 0.5      | face  |
| 21    | 2    | 0.5      | chair |


**TR**: supposing the TR = 1.5 s

**ER**: Effective resolution. This is useful when in your design the TR is not time-locked to the stimulus. In this case, the effective resolution can be shorter than TR.
For a randomly sampled TR desgin, you cannot know your effective resolution, though you can still specify a value of ER being submultiple of TR. For some other design
with TR more controlled, you can calculate the effective resolution (for example, TR = 1.5 s, and SOA = 2 s, if the TR is synchronized with any one of the stimuli, then the effective resolution = 0.5 s).
For our example, let's assume the ER = 0.5 s.

``` python
import hrf

# define the hrfretrieval object. The peristimulus window is set at -4 to 15 s relative to stimlus onset
ffa_hrfResponse = hrf.HrfRetrieval(tsData, Ev_file, TR=1.5, ER=0.5, window=[-4, 15])

# build up the FIR model
ffa_hrfResponse.getFIRmodel()

# get HRF for each condition. You can specify which method you'd like to use. 
# For "FIR", a FIR GLM approach will be implemented; 
# For "average", simple event-related response averaging will be implemented. 
# Note that for simple averaging, automatic baseline correction is implemented.
ffa_hrfResponse.estimateHRF(metric='average')

# the hrf can be read now. For example, for the condition of faces [event code: 0], the hemodynamic response is:
ffa_hrfResponse.cond[0]['average_hrf']

# if you use "FIR" approach, then the hrf can be read by:
ffa_hrfResponse.cond[0]['FIR_hrf']


```
