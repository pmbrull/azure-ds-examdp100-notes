# Build AI solutions with Azure Machine Learning service

## Introduction to Azure Machine Learning service

<details>
<summary> 
Show content
</summary>
<p>

### Learning Objectives

* Learn the difference between Azure Machine Learning Studio and Azure Machine Learning service
* See how Azure Machine Learning service fits into the data science process
* Learn the concepts related to an Azure Machine Learning service experiment
* Explore the Azure Machine Learning service pipeline
* Train a model using Azure Machine Learning service

### Azure Machine Learning Service within a data science process

Environment Set Up -> Data Preparation -> Experimentation -> Deployment

* **Environment setup**: First step is creating a **Workspace**, where you store your ML work. An **Experiment** is created within the workspace to store information about runs for your model. You can have multiple experiments in one workspace. You can interact with the environment with different IDEs such as PyCharm or Azure Notebooks.
* **Data Preparation**: explore, analyze and visualize the sources. You can use any tool. Azure provides the following SDK `Azureml.dataprep`.
* **Experimentation**: Iterative process of training and testing. With AMLS you can run the model in Azure containers. You need to create and configure a computer target object used to provision computer resources.
* **Deployment**: Create a Docker image that will get deployed to Azure Container Instances (you could also choose AKS, Azure IoT or FPGA).

### Create a machine learning experiment

![img](../assets/img/key-components-ml-workspace.png)

* **Workspace**: top-level resource in AMLS where you build and deploy your models. With a registered model and scoring scripts you can create an image for deployment. It stores experiment objects which save computer targets, track runs, logs, metrics and outputs.
* **Image**: it has three key components:
    1. A model and scoring script or application
    1. An environment file that declares the dependencies.
    1. A configuration file with the necessary resources to execute the model.
* **Datastore**: Abstraction over an Azure Storage account. Each workspace has a default one, but you could add Blob or File storage containers.
* **Pipeline**: Tool to create and manage workflows during a ds process. Each step can run unattended in different computer targets, which makes it easier to allocate resources.
* **Computer target**: Resource to run a training model or to host service deployment. It is attached to a workspace.
* **Deployed Web service**: You can choose between ACI, AKS or FPGA. With the model, script and image files you can create a Web service.
* **IoT module**: It is a Docker container and has the same needs as a Web Service. It enables to monitor a hosting device.

### Creating a pipeline

Some features or Azure ML pipelines are:
* Schedule tasks and executions,
* You can allocate different computer targets for different steps and coordinate multiple pipelines,
* You can reuse pipeline scripts and customize them,
* You can record and manage input, output, intermediate tasks and data.

### Knowledge Check

1. The Azure Machine Learning service SDK is which of the following?

* A visual machine learning development portal.
* A Python package containing functions to use the Azure ML service.
* A special type of Azure virtual machine.

    <details>
    <summary> 
    Answer
    </summary>
    <p>
    The modules provided by the Azure ML SDK provide the functions you need to work with the service in Python.
    </p>
    </details>

1. Which of the following is the underlying technology of the Azure Machine Learning service?

* Spark
* Hadoop
* Containerization including Docker and Kubernetes

    <details>
    <summary> 
    Answer
    </summary>
    <p>
    Containerization is a key technology used by the Azure ML service.
    </p>
    </details>

1. Which of the following is not a component of an Azure Machine Learning service workspace image?

* An R package
* An environment file that declares dependencies that are needed by the model, scoring script or application.
* A model scoring script

    <details>
    <summary> 
    Answer
    </summary>
    <p>
    R packages are not part of an Azure Machine Learning service workspace image.
    </p>
    </details>

1. Which of the following descriptions accurately describes Azure Machine Learning?

    * A Python library that you can use as an alternative to common machine learning frameworks like Scikit-Learn, PyTorch, and Tensorflow.
    * A cloud-based platform for operating machine learning solutions at scale.
    * An application for Microsoft Windows that enables you to create machine learning models by using a drag and drop interface.

    <details>
    <summary> 
    Answer
    </summary>
    <p>
    Cloud based Platform: Azure Machine Learning enables you to manage machine learning model data preparation, training, validation, and deployment. It supports existing frameworks such as Scikit-Learn, PyTorch, and Tensorflow; and provides a cross-platform platform for operationalizing machine learning in the cloud.
    </p>
    </details>

1. Which edition of Azure Machine Learning workspace should you provision if you only plan to use the graphical Designer tool to train machine learning models?

    * Basic
    * Enterprise

    <details>
    <summary> 
    Answer
    </summary>
    <p>
    The visual Designer tool is not available in Basic edition workspaces, so you must create an Enterprise workspace to use it.
    </p>
    </details>

1. You are using the Azure Machine Learning Python SDK to write code for an experiment. You must log metrics from each run of the experiment, and be able to retrieve them easily from each run. What should you do?

    * Add print statements to the experiment code to print the metrics.
    * Save the experiment data in the outputs folder.
    * Use the log* methods of the Run class to record named metrics.

    <details>
    <summary> 
    Answer
    </summary>
    <p>
    To record metrics in an experiment run, use the Run.log* methods.
    </p>
    </details>



</p>
</details>

---

## Train a local ML model with Azure Machine Learning service

<details>
<summary> 
Show content
</summary>
<p>

### Learning Objectives


* Use an Estimator to run a model training script as an Azure Machine Learning experiment.
* Create reusable, parameterized training scripts.
* Register models, including metadata such as performance metrics.

> As this is a rather practical module, you can refer to the labs notebooks or directly to Azure's docs.

### What is HyperDrive

HyperDrive is a built-in service that automatically launches multiple experiments in parallel each with different parameter configurations. Azure Machine Learning then automatically finds the configuration that results in the best performance measured by the metric you choose. The service will terminate poorly performing training runs to minimize compute resources usage.

### Azure Machine Learning estimators

In Azure Machine Learning, you can use a **Run Configuration** and a **Script Run Configuration** to run a script-based experiment that trains a machine learning model. However, these configurations may end up being really complex, so another abstraction layer is added: An **Estimator** encapsulates a run configuration and a script configuration in a single object.

We have some default Estimators for frameworks such as Scikit Learn, Pytorch and TF.

#### Writing a Script to Train a Model

After training a model, it should be saved in the **outputs** directory. For example witch SKlearn:

```python
from azureml.core import Run
import joblib

# Get the experiment run context
run = Run.get_context()

# Train and test...

# Save the trained model
os.makedirs('outputs', exist_ok=True)
joblib.dump(value=model, filename='outputs/model.pkl')

run.complete()
```

#### Using an Estimator

You can use a generic Estimator class to define a run configuration for a training script like this:

```python
from azureml.train.estimator import Estimator
from azureml.core import Experiment

# Create an estimator
estimator = Estimator(source_directory='experiment_folder',
                      entry_script='training_script.py',
                      compute_target='local',
                      conda_packages=['scikit-learn']
                      )

# Or use a framework specific estimator as
estimator = SKLearn(source_directory='experiment_folder',
                    entry_script='training_script.py'
                    compute_target='local'
                    )

# Create and run an experiment
experiment = Experiment(workspace = ws, name = 'training_experiment')
run = experiment.submit(config=estimator)
```

### Using script parameters

Used to increase the flexibility of script-based experiments.

These parameters are read as usual Python parameters in scripts. So for example, after setting the `Run`:

```python
# Set regularization hyperparameter
parser = argparse.ArgumentParser()
parser.add_argument('--reg_rate', type=float, dest='reg', default=0.01)
args = parser.parse_args()
reg = args.reg
```

To use parameters in **Estimators**, add the `script_params` value as a dict:

```python
# Create an estimator
estimator = SKLearn(source_directory='experiment_folder',
                    entry_script='training_script.py',
                    script_params = {'--reg_rate': 0.1},
                    compute_target='local'
                    )
```

### Registering models

After running an experiment that trains a model you can use a reference to the Run object to retrieve its outputs, including the trained model.

#### Retrieving Model Files

From the `run` object we can get all the files that it generated with `run.get_file_names()` and download the models as (recall how we said that usually those were stored under `outputs/`)

```python
run.download_file(name='outputs/model.pkl', output_file_path='model.pkl')
```

#### Registering a Model

With `Model.register()` we can save different versions of our models:

```python
from azureml.core import Model

model = Model.register(workspace=ws,
                       model_name='classification_model',
                       model_path='model.pkl', # local path
                       description='A classification model',
                       tags={'dept': 'sales'},
                       model_framework=Model.Framework.SCIKITLEARN,
                       model_framework_version='0.20.3')
```

Or the same by referencing the `run` object:

```python
run.register_model( model_name='classification_model',
                    model_path='outputs/model.pkl', # run outputs path
                    description='A classification model',
                    tags={'dept': 'sales'},
                    model_framework=Model.Framework.SCIKITLEARN,
                    model_framework_version='0.20.3')
```

We can then view all the models we saved by using:

```python
for model in Model.list(ws):
    # Get model name and auto-generated version
    print(model.name, 'version:', model.version)
```

### Knowledge Check

1. An Experiment contains which of the following?

   * A composition of a series of runs
   * A Docker image
   * The data used for model training


    <details>
    <summary> 
    Answer
    </summary>
    <p>
    A composition of a series of runs: Azure ML Studio provides a visual drag and drop machine learning development portal but that is a separate offering.
    </p>
    </details>


1. A run refers to which of the following?

   * Python code for a specific task such as training a model or tuning hyperparameters. Run does the job of logging metrics and uploading the results to Azure platform.
   * A set of containers managed by Kubertes to run your models.
   * A Spark cluster.



    <details>
    <summary> 
    Answer
    </summary>
    <p>
    Python code for a specific task such as training a model or tuning hyperparameters. Run does the job of logging metrics and uploading the results to Azure platform. 
    </p>
    </details>


1. A hyperparameter is which of the following?

   * A model parameter that cannot be learned by the model training process.
   * A model feature derived from the source data.
   * A parameter that automatically and frequently changes value during a single model training run.



    <details>
    <summary> 
    Answer
    </summary>
    <p>
    Hyperparameters control how the model training executes and must be set before model training.
    </p>
    </details>


1. Before you can train and run experiments in your code, you must do which of the following?

   * Create a virtual machine
   * Log out of the Azure portal
   * Write a model scoring script


    <details>
    <summary> 
    Answer
    </summary>
    <p>
    Your Python script needs to connect to the Azure ML workspace before you can train and run experiments.
    </p>
    </details>


1. Which of the following is a technique for determining hyperparameter values?

   * grid searching
   * Bayesian sampling
   * hyper searching



    <details>
    <summary> 
    Answer
    </summary>
    <p>
    Grid searching is often used by data scientists to find the best hyperparamter value.
    </p>
    </details>

1. You have written a script that uses the Scikit-Learn framework to train a model. Which framework-specific estimator should you use to run the script as an experiment?

    * PyTorch
    * Tensorflow
    * SKLearn


    <details>
    <summary> 
    Answer
    </summary>
    <p>
    To run a scikit-learn training script as an experiment, use the generic Estimator estimator or a SKLearn estimator.
    </p>
    </details>


1. You have run an experiment to train a model. You want the model to be stored in the workspace, and available to other experiments and published services. What should you do?

   * Register the model in the workspace.
   * Save the model as a file in a Compute Instance.
   * Save the experiment script as a notebook.

    <details>
    <summary> 
    Answer
    </summary>
    <p>
    To store a model in the workspace, register it.
    </p>
    </details>

</p>
</details>

---


## Working with Data in Azure Machine Learning

<details>
<summary> 
Show content
</summary>
<p>

### Learning objectives

* Create and use datastores
* Create and use datasets

