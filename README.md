# Music-Genre-Classification

Music-Genre-Classification is a project I realized in the first semester of my Master's studies within the Artificial Neural Networks course at the Wroclaw University of Technology in the field of Control Engineering and Robotics. The main goal of this task was to get acquainted with the processing of a selected dataset and to create a neural network model that allows to identify a music genre based on a sound signal.

Dataset used for this project: [GTZAN Dataset - Music Genre Classification](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification)

## Setup
The scripts were tested for `Python 3.10.6` version.

Create virtual environment:

```
python3 -m venv venv
```

Activate virtual environment:

```
source venv/bin/activate 
```

Install requirements:

```
pip3 install -r requirements.txt
```

## Usage

To predict the music genre of a selected music file, run the `predict_genre.py`

```
python3 predict_genre.py
usage: predict_genre.py [-h] filename iterations
predict_genre.py: error: the following arguments are required: filename, iterations
```

Example usage:

```
python3 predict_genre.py song.mp3 50
```