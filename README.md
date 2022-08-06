# JASA_EL_LEV
This repository hosts python code and data of publication 'Surrounding line sources optimally reproduce diffuse envelopment at off-center listening positions'. In the paper we show that for the horizontal layer of a surround sound system, line sources are preferable over the typical point-source loudspeakers.

Dependencies:
* numpy
* scipy
* matplotlib
* joblib
  
  
*plotListeningArea_IC_ILD.py*:  
plots interaural cues across a simulated listening area for multi-source arrangements. 

*plotExperimentResults.py*:  
plots statistics of experiment results with visualizations of conditions.  
  
*plotFreqDependent_IC_ILD.py*:  
plots interaural coherence and level difference over frequency for specified conditions of the listening experiment.  
  
*plotAngularError_IC_ILD.py*:  
plots interaural coherence and level difference over frequency, showing influence of angular error of the discrete remapping used in experiment.  
  
All plots are directly saved to the 'Figures' subfolder without pop-up figure windows.

In the 'Figures' directory you can find many pre-rendered plots. The following plot shows that line sources provide the largest area of optimal cues for envelopment: a low interaural level difference (< 1dB) and a low interaural coherence (< 0.4). Here we assume the loudspeakers to play incoherent sound signals (diffuse-field reproduction).

<img src="/Figures/ListeningArea_IC_ILD/12LS_sweet_area_ERB.jpg" alt="drawing" width="500"/>

Here is an example of a rectangular loudspeaker setup, which is typical for venues like movie theaters:
<img src="/Figures/ListeningArea_IC_ILD/RECT_80wide100long_16LS_sweet_area_ERB.jpg" alt="drawing" width="500"/>
