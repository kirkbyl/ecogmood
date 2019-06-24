# ecogmood

Code repository for *An Amygdala-Hippocampus Subnetwork that Encodes Variation in Human Mood*, L.A.Kirkby et al., Cell (2018)

DOI: https://doi.org/10.1016/j.cell.2018.10.005

**Study goal:** to identify intrinsic neural networks that encode variation in human mood using large-scale intracranial EEG recordings.

**Repository organization:**
* *ecogtools*: general tools for preprocessing and analysis of ecog datasets
* *projmods*: project-specific modules and analysis classes

** *Note:* ** analysis framework is built upon creating Patient data objects (refer to *ecogtools.recordingparams.subjects.Patient*). This requires access to raw datasets, however, these cannot currently be provided due to patient consent considertations. Therefore, to make use of analysis code, Patient class must be adapted and tailored to specific user dataset.


**Analysis overview:**
1. Preprocess raw ECoG signals:
* Band-pass filter signals between 0.5–256 Hz and downsample to 512 Hz using 8th order chebyshev type I filter
* Notch-filter at 60, 120, 180 and 240 Hz with 4 Hz bandwidth using 5th order butterworth filter
* Re-reference to the common average across channels sharing the same lead

*Refer to functions in ecogtools.preprocess.filters*

2. Generate coherence matrices:
* Split voltage traces from electrodes located in brain regions of interest into contiguous 10s segments
* Calculate signal coherence between all pairs of electrodes for each 10s segment. Power spectral density computed using Welch’s method with a non-overlapping Hanning window
* Repeat coherence calculation using phase randomized surrogate signals (signals with the same power spectra as the original signals but reconstructed with randomized phases). Phase randomized values are subtracted from the coherence of the original signal
* Construct time-series of coherence matrices by averaging across chosen frequency bands: theta θ, [4-8 Hz]; alpha, α [8-13 Hz]; beta, β [13-30 Hz]; and, gamma γ [30-70 Hz]

*Refer to functions in ecogtools.analysis.coherence*

3. Generate Intrinsic Coherence Networks (ICNs) for each subject
* Run PCA on each coherence matrix
* Estimate number of significant components using Marchenko-Pastur Law (Lopes-dos-Santos et al., 2013)
* ICA carried out on the significant PCs to separate the signal mixtures into independent sources
* Visualize ICNs by plotting chord diagrams

*Refer to functions in ecogtools.analysis.coherence for ICN generation and in projmods.icnetworks for ICN visualization*

4. Topological clustering to identify groups of similar ICNs across subjects
* Construct similarity and adjacency matrices by computing correlation coefficient between all pairs of ICNs
* Construct topological map to show connected nodes/similiar ICNs

*Refer to functions in projmods.icnetworks.icnclusters*

5. Construct linear model to predict IMS (immediate mood score) from ICNs
* Use eleastic net linear model (uses both L1 and L2 regularization parameters) to predict IMS from ICN features separately for each subject 
* Assess model significance by comparing true vs. shuffled datasets
* Identify common mood-related neural ICN activity feature across subjects
* Regress IMS against mood-related neural ICN activity feature (z-score normalized) to pool data across subjects 

*Refer to functions in projmods.icnmodel for model fitting and in projmods.imsbiomarker for pooling across subjects*

