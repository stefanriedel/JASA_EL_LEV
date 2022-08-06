# JASA_EL_LEV
This repository hosts python code and data of publication 'Surrounding line sources optimally reproduce diffuse envelopment at off-center listening positions'.

Dependencies:
* numpy
* scipy
* matplotlib
* joblib
  
  
*plotExperimentResults.py*:  
plots statistics of experiment results with visualizations of conditions.  
  
*plotGammatone_IC_ILD.py*:  
plots interaural coherence and level difference over frequency for specified conditions of the experiment.  
  
*sweetAreaLEVBinaural.py*:  
plots interaural cues across a simulated listening area for multi-source arrangements.    
  
All plots are directly saved to the 'Figures' subfolder without pop-up figure windows.


In the Figures folder you can find many prerendered plots of circular and rectangular arrangements, showing that line sources provide the largest area of optimal cues for envelopment: a low interaural level difference and a low interaural coherence. Here an example for 12 loudspeakers in a ring:

![alt text](/Figures/12LS_sweet_area_ERB#.jpg)

<img src="/Figures/12LS_sweet_area_ERB.jpg" alt="drawing" width="500"/>

Here an example of a rectangular setup, as is typical for cinema sound systems:

<img src="/Figures/RECT_80wide100long_16LS_sweet_area_ERB.jpg" alt="drawing" width="500"/>


![alt text](/Figures/RECT_80wide100long_16LS_sweet_area_ERB#.jpg)

