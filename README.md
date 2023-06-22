# VIVIT Spatiotemporal for Hierarchical Anomaly Detection in Video Scenes

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)
![VIVIT](https://img.shields.io/badge/VIVIT-Spatiotemporal-brightgreen.svg?style=for-the-badge)

![Python](https://img.shields.io/badge/Python-3.9-blue.svg?style=for-the-badge)
![TensorFlow](https://img.shields.io/badge/TensorFlow-v2.7-orange.svg?style=for-the-badge)
![Keras](https://img.shields.io/badge/Keras-v2.8.0-red.svg?style=for-the-badge)

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/moharamfatema/graduation-project/HEAD)

This repository contains Colab notebooks for the VIVIT spatiotemporal model, which focuses on hierarchical anomaly detection in video scenes. The project includes various stages such as data processing, model implementation in the Keras TensorFlow framework, hyperparameter tuning, cross-testing, and transfer learning.

## Project Overview

The VIVIT spatiotemporal model aims to detect anomalies in video scenes by leveraging hierarchical representations. The project is divided into several stages, each of which is covered in a separate Colab notebook:

1. [Data Processing](./data-preprocessing): This folder covers the preprocessing and preparation of the video datasets for training and evaluation. It includes steps such as video loading, frame extraction, normalization, and data augmentation.

1. [Hyperparameter Tuning](./hyperparameter-tuning): This folder focuses on optimizing the model's hyperparameters to achieve the best performance. It explores different hyperparameter configurations and uses techniques like hyperband to find the optimal settings.

1. [Transfer Learning](./transfer-learning): This folder covers the transfer learning stage, where the model is trained on a source dataset and then fine-tuned on a target dataset. It includes steps such as freezing layers, adjusting the learning rate, and training the model.

1. [Weapon Detection](./weapon-detection): This folder focuses on the weapon detection stage, where the model is trained to detect weapons in video scenes.

1. [Spatio Temporal Model](./spatiotemporal): This folder covers the implementation of the VIVIT spatiotemporal model. It includes steps such as data loading, model implementation, and training.

1. [CNN Replication](./cnn-replication): This folder focuses on replicating the CNN model to compare its performance with the VIVIT spatiotemporal model. It includes steps such as data loading, model implementation, and training.

## Other Repositories

To complement this repository, you can also find the code for the backend and frontend of the project in separate repositories:

- Backend Repository: [grad-be](https://github.com/moharamfatema/grad-be)
- Frontend Repository: [grad-fe](https://github.com/moharamfatema/grad-fe)

Please refer to these repositories for the backend and frontend code related to the website.

## Getting Started

To use the Colab notebooks in this repository, follow these steps:

1. Clone this repository to your local machine using the following command:

```
git clone https://github.com/moharamfatema/graduation-project.git
```

2. Open the desired notebook in Google Colab or Jupyter Notebook to explore the project stages.

3. Follow the instructions and comments within each notebook to execute the code, preprocess data, implement the model, tune hyperparameters, cross-test, or apply transfer learning.

Feel free to explore the notebooks, modify the code, and adapt the project to your specific needs.

## License

This project is licensed under the [MIT License](LICENSE).
