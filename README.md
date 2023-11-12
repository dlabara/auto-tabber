auto-tabber
==============================

This repository started out as a summer project with the goal of putting into practice my ML and Python skills. auto-tabber is a drum tab creator driven by Deep Neural Networks (DNNs), similarly to https://github.com/mayhemsloth/Drum-Tabber, although with a different approach. 

By analyzing a database of Guitar Pro files and corresponding MP3s, the neural network learns to identify diverse drum hits. The goal of this project is to have a end to end tab creator, inputing the .mp3 file, and obtaining the MIDI or .gp5 file as an output. This would automate the drum tabbing process, saving time for musicians. The concept could also be used to create tabs for other instruments in the future, but drums are probably the easiest starting point.

The state of the project is very preliminary, a lot of work was done on the preprocessing of the raw .gp5 files for the training data. The current architecture seems to be working fine with the limited songs I have in the dataset. Currently no MIDI or .gp5 files are created but the output from the NN and the postprocessing is viewed through a simple dot graph.

There are a lot of areas where work is needed:
- Extending data read capability to more file formats (.gp4, .gp3)
- Refactoring the code to adhere to the cookiecutter template
- Automating the train, test, validate, and try phases of the model
- Extending the database with a larger database of songs (matched Guitar Pro and .mp3 files)
- Implementing a data batching process for the model training (I am running out of RAM/VRAM in my platform)
- Trying out different model architectures (I started out using a LSTM NN due to this being fundamentally an audio recognition problem), maybe CNNs or other Networks can work fine.
- Hypertunning the model parameters

I currently don't have a lot of time to put into the project (only the weekends), but I am more than happy to collaborate with anyone interested, but keep in mind that I take this as a learning exercise.

Take a look and have fun!

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
