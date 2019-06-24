# ECoG data analysis package

### Code to preprocess, analyze and visualize data from human ECoG recordings

#### Directory structure:

```
|--- README.md
|
|--- __init__.py
|
|---preprocess                  <-- Code related to signal preprocessing
|   |
|   |--__init__.py
|   |--h5IO.py                  <-- Input/output of data from hdf5 files
|   |--filters.py               <-- Spectral and spatial filters
|
|--analysis                     <-- Code related to general analysis
|   |
|   |--__init__.py
|   |--coherences.py            <-- Computation and analysis of signal coherence
|   |--IMScorr.py               <-- Analysis of correlation of immediate mood scale with neural features
|   |--models.py                <-- Model fitting
|
|--recordingparams              <-- Code related to data-/recording-specific information
|   |
|   |--__init__.py
|   |--elecs.py                 <-- Electrode locations, frequency components etc
|   |--psych.py                 <-- Neuropsych info eg. IMS mood data, depression and anxiety indices
|   |--subjects.py              <-- Subject-specific info eg. subnets patients
|
|--tools                        <-- General helper functions/utilities
|   |
|   |--__init__.py
|   |--loaddata.py              <-- Loading data or converting loaded data in one format into another
|   |--paths.py                 <-- Pointers to useful paths
|   |--utilities.py             <-- Misc helpful functions
|
|--visualization                <-- Code related to data visualization
|   |   
|   |--__init__.py
|   |--plotcoherence.py         <-- Plotting coherence matrices, PCA/ICA/ICNs
|   |--plotcorrelations.py      <-- Regression plots
|   |--plotchord.py             <-- Plotting chord connectivity diagrams
```
