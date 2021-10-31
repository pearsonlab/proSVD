# Procrustean SVD (proSVD)
Tools for streaming dimension reduction analyses of neural data.

See `notebooks/` for usage.

#### To build:
```
$ git clone https://github.com/pearsonlab/streamingSVD
$ cd streamingSVD
$ pip install .
```

#### TODO:
- check correctness of init with with some other basis
  - use this to determine the alignment of some "hypothesis" subspace with the data?
- artefact rejection with online ICA (ORICA)?
- add input functions to pro.run() to run before (e.g. preprocessing) and after (e.g. project onto new subspace) each update 
- do we care about W???
- rename repo to proSVD

#### datasets
- iEEG seizures
- EEG artefacts (TMS or eyeblink)
- monkey reaching (O'shea data, jPCA data)
- Neuropixels (Stringer, Musall)
- widefield/behavior? (both pretty large data)

#### methods
- online:
  - Baker's seq-kl
  - slow feature analysis
  - ica
  - jpca
- offline:
  - SVD
  - PSID?