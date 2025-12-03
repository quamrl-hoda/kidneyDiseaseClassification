# Kidney Disease Classification DL Project
### install the requirements
```bash
pip install - requirements.txt
```

# workflows
- Update config.yaml
- Update secrets.yaml [Optional]
- Update params.yaml
- Update the entity
- Update the configuration manager in src config
- Update the components
- Update the pipeline
- Update the main.py
- Update the dvc.yaml
- app.py


#### cmd
- mlfow ui


### dagshub
[dagshub](https://dagshub.com/)

os.environ["MLFLOW_TRACKING_URI"] = hps://dagshub.com/quamrl-hoda/kidneyDiseaseClassificationDeepLearningProject.mlflow \
os.environ["MLFLOW_TRACKING_USERNAME"] = quamrl-hoda \
os.environ["MLFLOW_TRACKING_PASSWORD"] = 10bd4471835788590657bbf53e006492e901bc34 \
python scripts.py

Run this to export as env variables:

```bash

set MLFLOW_TRACKING_URI = https://dagshub.com/quamrl-hoda/kidneyDiseaseClassificationDeepLearningProject.mlflow

set MLFLOW_TRACKING_USERNAME = quamrl-hoda

set MLFLOW_TRACKING_PASSWORD = 10bd4471835788590657bbf53e006492e901bc34

```


### DVC cmd

- dvc init
- dvc repro
- dvc dag