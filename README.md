# bioactivity-prediction-app

# Reproducing this web app
To recreate this web app on your own computer, do the following.

### Create conda environment
Firstly, we will create a conda environment called *bioactivity*
```
conda create -n bioactivity python=3.7.9
```
Secondly, we will login to the *bioactivity* environement
```
conda activate bioactivity
```
### Install prerequisite libraries

Download requirements.txt file

```
wget https://raw.githubusercontent.com/dataprofessor/bioactivity-prediction-app/main/requirements.txt

```

Pip install libraries
```
pip install -r requirements.txt
```

###  Launch the app

```
streamlit run app.py
```
