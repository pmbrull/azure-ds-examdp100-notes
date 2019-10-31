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

</p>
</details>

---

