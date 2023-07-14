## <a name="getting-ascad"> Getting the ASCAD databases and the trained models 

In the new folder, download the data packages with the raw data by using:

<pre>
$ cd ASCAD/ATMEGA_AES_v1/ATM_AES_v1_variable_key/
$ mkdir -p ASCAD_data/ASCAD_databases
$ cd ASCAD_data/ASCAD_databases
$ wget https://static.data.gouv.fr/resources/ascad-atmega-8515-variable-key/20190730-071646/atmega8515-raw-traces.h5
$ mv atmega8515-raw-traces.h5 ATMega8515_raw_traces.h5
$ wget https://static.data.gouv.fr/resources/ascad-atmega-8515-variable-key/20190903-083349/ascad-variable.h5
$ mv ascad-variable.h5 ASCAD.h5
$ wget https://static.data.gouv.fr/resources/ascad-atmega-8515-variable-key/20190903-084119/ascad-variable-desync50.h5
$ mv ascad-variable-desync50.h5 ASCAD_desync50.h5
$ wget https://static.data.gouv.fr/resources/ascad-atmega-8515-variable-key/20190903-084306/ascad-variable-desync100.h5
$ mv ascad-variable-desync100.h5 ASCAD_desync100.h5
</pre>

Please be aware that all these steps should **download around 71 GB** of data.
You can selectively download only the extracted databases (`https://static.data.gouv.fr/resources/ascad-atmega-8515-variable-key/20190903-083349/ascad-variable.h5`
and so on) that weight a more reasonable 418 MB each.

### Raw data files hashes

The data files SHA-256 hash values are:

<hr>

**ASCAD/ATMEGA_AES_v1/ATM_AES_v1_variable_key/ASCAD_data/ASCAD_databases/ASCAD.h5:**
`d834da6ca5a288c4ba5add8e336845270a055d6eaf854dcf2f325a2eb6d7de06`
**ASCAD/ATMEGA_AES_v1/ATM_AES_v1_variable_key/ASCAD_data/ASCAD_databases/ASCAD_desync50.h5:**
`0fa048bf42b9d8bbf9514770072c070175637e1b6fb6da370e2b020b1ecca673`
**ASCAD/ATMEGA_AES_v1/ATM_AES_v1_variable_key/ASCAD_data/ASCAD_databases/ASCAD_desync100.h5:**
`cb82c553b84c29454ea23ec043730ed845e17c2a0e261853afc11e028fdf2710`
**ASCAD/ATMEGA_AES_v1/ATM_AES_v1_variable_key/ASCAD_data/ASCAD_databases/ATMega8515_raw_traces.h5:**
`6f13d7c380c937892c09b439910c4313d551adf011d2f4d76ad8b9b3de27b852`

> **WARNING: all the paths and examples that are provided below suppose that you have
downloaded and decompressed the raw data file as explained in [the previous section](#getting-ascad).**

<hr>

## The ATMega8515 SCA traces databases

This database contains 300,000 traces from the acquisition campaign compiled in a [HDF5](https://en.wikipedia.org/wiki/Hierarchical_Data_Format) file of 71 GB named `ATMega8515_raw_traces.h5`. The structure of this HDF5 file is described in the article ["Study of Deep Learning Techniques for Side-Channel Analysis and Introduction to ASCAD Database"](https://eprint.iacr.org/2018/053.pdf).

We emphasize that these traces are **not synchronized**. The key is **variable** in the following fashion: one acquisition every three uses a **fixed** key and the other two out of three use a **random** key.


## The ASCAD databases

The databases, which are HDF5 files, basically contain two labeled datasets:
  * A 200,000 traces **profiling dataset** that is used to train the (deep) Neural Networks models.
  * A 100,000  traces **attack dataset** that is used to check the performance of the trained models after the
profiling phase. 

The ASCAD database is in fact extracted from the `ATMega8515_raw_traces.h5` file containing raw traces: in order to avoid useless heavy data processing, a window of 1400 points of
interest are extracted around the leaking spot.

The [ASCAD_generate.py](ASCAD_generate.py) script is used to generate ASCAD from the ATMega8515 original database. Actually, 
the repository contains three HDF5 ASCAD databases:

  * `ASCAD_data/ASCAD_databases/ASCAD.h5`: this corresponds to 
    the original traces extracted without modification.
  * `ASCAD_data/ASCAD_databases/ASCAD_desync50.h5`: this
    contains traces desynchronized with a 50 samples maximum window.
  * `ASCAD_data/ASCAD_databases/ASCAD_desync100.h5`: this
    contains traces desynchronized with a 100 samples maximum window.

You can generate new databases with random desynchronization using the [ASCAD_generate.py](ASCAD_generate.py) script.

## The trained models

The best **trained CNN models** that we have obtained are provided and can be downloaded using the following URLs:

<pre>
$ wget https://static.data.gouv.fr/resources/ascad-atmega-8515-variable-key/20190801-132322/cnn2-ascad-desync0.h5
$ wget https://static.data.gouv.fr/resources/ascad-atmega-8515-variable-key/20190801-132406/cnn2-ascad-desync50.h5
</pre>


Two models have been selected: best CNN for desynchronizations 0 (`cnn2-ascad-desync0.h5`) and best CNN for desynchronizations 50 (`https://static.data.gouv.fr/resources/ascad-atmega-8515-variable-key/20190801-132406/cnn2-ascad-desync50.h5`).


**WARNING**: these models are the best ones we have obtained through the methodology described in the article. We certainly **do not pretend** that they are the optimal models  across all the possible ones. The main purpose of sharing ASCAD is precisely to explore and evaluate new models.
