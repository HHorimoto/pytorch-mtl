# Multi Task Learning (MTL) with Pytorch

## Usage

```bash
$ git clone git@github.com:HHorimoto/pytorch-mtl.git
$ cd pytorch-mtl
$ ~/python3.10/bin/python3 -m venv .venv
$ . .venv/bin/activate
$ pip install -r requirements.txt
$ source run.sh
```

## Features

### Multi Task Learning  
I trained the model for 50 epochs under both single-task learning (STL) and multi-task learning (MTL) scenarios.

* The main task is a 10-class classification using CIFAR-10.

* The subtask is a binary classification: distinguishing between animals (bird, cat, deer, dog, horse, frog) and non-animals (airplane, automobile, ship, truck).

The table below presents the results obtained from the experiments.

**Accuracy Comparison**

|     | Accuracy |
| --- | :------: |
| STL |  0.5448  |
| MTL |  0.5636  |


#### Reference
[1] [https://medium.com/@aminul.huq11/multi-task-learning-a-beginners-guide-a1fc17808688]

[2] [https://github.com/machine-perception-robotics-group/MPRGDeepLearningLectureNotebook/blob/master/11_cnn_pytorch/09_multitask_fundamental.ipynb]