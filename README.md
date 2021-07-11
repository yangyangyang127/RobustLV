# Code for "Robustness Enhanced Quantification of LeftVentricle via Directional Subbands Decomposition"

This is an end-to-end framework for accurate and robust left ventricleindices quantification, including cavity and myocardiumareas, six regional wall thicknesses, 
and three directionaldimensions.  

The proposed method first decomposes a cardiovascular magnetic resonance image into directionalfrequency bands via Steerable Pyramid Transformation.Then deep representations of 
each direction are extractedseparately via a CNN model and the temporal correlation between frames were modeled with a recurrent neural net-work. 
Finally, we explore the multidirectional relationship offeatures, indices, and directional subbands to optimize thequantification  system. 

The dataset we used can be found at MICCAI 2018 Left Ventricle Full Quantification Challenge.



