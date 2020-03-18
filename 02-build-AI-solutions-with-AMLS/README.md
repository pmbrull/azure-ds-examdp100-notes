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

### Introduction to datastores

Abstractions for cloud data sources. They hold the connection information and can be used to both read and write. The different sources could be (sample from [here](https://docs.microsoft.com/en-us/azure/machine-learning/concept-data#access-data-in-storage)):

* Azure Storage (blob and file containers)
* Azure Data Lake Storage
* Azure SQL Database
* Azure Databricks file system (DBFS)

#### Using datastores

Each workspace has two built-in datastores (blob container + Azure Storage File container) used as system storage by AMLS. You have a limited use on top of those.

The good part of using external datasources - which is the usual - is the ability to share data accross multiple experiments, regardless of the compute context in which those experiments are running.

You can use the AMLS SDK to store / retrieve data from the datastores.

#### Registering a datastore

To register a datastore, you could either use the UI in AMLS or the SDK:

```python
from azureml.core import Workspace, Datastore

ws = Workspace.from_config()

# Register a new datastore
blob_ds = Datastore.register_azure_blob_container(
    workspace=ws,
    datastore_name='blob_data',
    container_name='data_container',
    account_name='az_store_acct',
    account_key='123456abcde789…'
)
```

#### Managing datastores

Again, managing can be done via UI or SDK:

```python
# list
for ds_name in ws.datastores:
    print(ds_name)

# get
blob_store = Datastore.get(ws, datastore_name='blob_data')

# get default
default_store = ws.get_default_datastore()

# set default
ws.set_default_datastore('blob_data')
```

### Use datastores

You can interact directly with a datastore via the SDK and *pass data references* to scripts that need to access data.

> OBS: For blobs to work correctly as a datastore and be accessible in the code to upload / download, the storage account should be Standard / Hot, not Premium!

#### Working directly with a datastore

```python
blob_ds.upload(src_dir='/files',
               target_path='/data/files',
               overwrite=True, show_progress=True)

blob_ds.download(target_path='downloads',
                 prefix='/data',
                 show_progress=True)
```

#### Using data references

When you want to use a datastore in an experiment script, you must pass a data reference to the script. There are the following accesses:

* **Download**: Contents are downloaded to the compute context.
* **Upload**: The files generated by the experiment are uploaded to the datastore after the run completes.
* **Mount**: When experiments run on a remote compute (not local), you can mount the path.

To pass the reference to an experiment script, define the `script_params`:

```python
data_ref = blob_ds.path('data/files').as_download(path_on_compute='training_data')
estimator = SKLearn(source_directory='experiment_folder',
                    entry_script='training_script.py'
                    compute_target='local',
                    script_params = {'--data_folder': data_ref})
```

`script_params` can then be retrieved via `argparse`.

### Introduction to datasets

Datasets are versioned packaged data objects that can be easily consumed in experiments and pipelines. They are the recommended way to work with data.

Datasets can be based on files in a datastore or on URLs and other resources.

#### Types of dataset

* **Tabular**: useful when when we work, for example, with pandas.
* **File**: For unstructured data. Dataset will present a list of paths that can be read as thought from the file system. For example, for images in a CNN.

#### Creating and registering datasets

You can use the UI or the SDK to create datasets from files or paths (which can include wildcards `*` for regex).

##### Creating and registering tabular datasets

```python
from azureml.core import Dataset

blob_ds = ws.get_default_datastore()
csv_paths = [(blob_ds, 'data/files/current_data.csv'),
             (blob_ds, 'data/files/archive/*.csv')]
tab_ds = Dataset.Tabular.from_delimited_files(path=csv_paths)
tab_ds = tab_ds.register(workspace=ws, name='csv_table')
```

##### Creating and registering file datasets

```python
from azureml.core import Dataset

blob_ds = ws.get_default_datastore()
file_ds = Dataset.File.from_files(path=(blob_ds, 'data/files/images/*.jpg'))
file_ds = file_ds.register(workspace=ws, name='img_files')
```

#### Retrieving a registered dataset

You can retrieve datasets by the `datasets` attribute of a `Workspace` or by calling `get_by_name` or `get_by_id` of the `Dataset` class:

```python
import azureml.core
from azureml.core import Workspace, Dataset

# Load the workspace from the saved config file
ws = Workspace.from_config()

# Get a dataset from the workspace datasets collection
ds1 = ws.datasets['csv_table']

# Get a dataset by name from the datasets class
ds2 = Dataset.get_by_name(ws, 'img_files')
```

#### Dataset versioning

Useful to reproduce experiments with data in the same state. Use the `create_new_version` property when registering a dataset:

```python
img_paths = [(blob_ds, 'data/files/images/*.jpg'),
             (blob_ds, 'data/files/images/*.png')]
file_ds = Dataset.File.from_files(path=img_paths)
file_ds = file_ds.register(workspace=ws, name='img_files', create_new_version=True)
```

To retrieve a specific version:

```python
img_ds = Dataset.get_by_name(workspace=ws, name='img_files', version=2)
```

### Use datasets

You can read data directly from a dataset, or you can pass a dataset as a named input to a script configuration or estimator.

#### Working with a dataset directly

If you have a reference to a dataset, you can access its contents directly.

```python
df = tab_ds.to_pandas_dataframe()
```

When working with a file dataset, use `to_path()`:

```python
for file_path in file_ds.to_path():
    print(file_path)
```

#### Passing a dataset to an experiment script

When you need to access a dataset in an experiment script, you can pass the dataset as an input to a **ScriptRunConfig** or an **Estimator**:

```python
estimator = SKLearn( source_directory='experiment_folder',
                     entry_script='training_script.py',
                     compute_target='local',
                     inputs=[tab_ds.as_named_input('csv_data')],
                     pip_packages=['azureml-dataprep[pandas]')
```

Since the script will need to work with a **Dataset** object, you must include either the full **azureml-sdk** package or the **azureml-dataprep** package with the **pandas** extra library in the script's compute environment.

Then, in the experiment

```python
run = Run.get_context()
data = run.input_datasets['csv_data'].to_pandas_dataframe()
```

Finally, when passing a file dataset, you must specify the access mode:

```python
estimator = Estimator( source_directory='experiment_folder',
                     entry_script='training_script.py'
                     compute_target='local',
                     inputs=[img_ds.as_named_input('img_data').as_download(path_on_compute='data')],
                     pip_packages=['azureml-dataprep[pandas]')
```

### Knowledge Check

1. You've uploaded some data files to a folder in a blob container, and registered the blob container as a datastore in your Azure Machine Learning workspace. You want to run a script as an experiment that loads the data files and trains a model. What should you do?

   * Save the experiment script in the same blob folder as the data files.
   * Create a data reference for the datastore location and pass it to the script as a parameter.
   * Create global variables for the Azure Storage account name and key in the experiment script.

    <details>
    <summary> 
    Answer
    </summary>
    <p>
    To access a path in a datastore in an experiment script, you must create a data reference and pass it to the script as a parameter. The script can then read data from the data reference parameter just like a local file path.
    </p>
    </details>

1. You've registered a dataset in your workspace. You want to use the dataset in an experiment script that is run using an estimator. What should you do?

   * Pass the dataset as a named input to the estimator.
   * Create a data reference for the datastore location where the dataset data is stored, and pass it to the script as a parameter.
   * Use the dataset to save the data as a CSV file in the experiment script folder before running the experiment.

    <details>
    <summary> 
    Answer
    </summary>
    <p>
    To access a dataset in an experiment script, pass the dataset as a named input to the estimator. 
    </p>
    </details>

</p>
</details>

---


## Working with Compute Contexts in Azure Machine Learning

<details>
<summary> 
Show content
</summary>
<p>

### Learning objectives

* Create and use environments.
* Create and use compute targets.

### Introduction to environments

Python code runs in the context of a virtual environment that defines the version of the Python runtime to be used as well as the installed packages available to the code.

#### Environments in Azure Machine Learning

In general, AML handles environment creationm, package installation and environment registration for you - usually through the creation of Docker containers. You'd just need to specify the packages you want. You could also manage the environments if needed.

Environments are encapsulated by the **Environment** class; which you can use to create environments and specify runtime configuration for an experiment.

#### Creating environments

* **Creating an environment from a specification file**: based on conda or pip. For example, a file named **conda.yml**
  
    ```
    name: py_env
        dependencies:
        - numpy
        - pandas
        - scikit-learn
        - pip:
            - azureml-defaults
   ```

   Then, create the environment with the SDK

   ```python
    from azureml.core import Environment

    env = Environment.from_conda_specification(name='training_environment',
                                            file_path='./conda.yml')
   ```

* **Creating an environment from an existing Conda environment**: If you have already a defined Conda environment on the workstation you can reuse it in AML

    ```python
    from azureml.core import Environment

    env = Environment.from_existing_conda_environment(name='training_environment',
                                                    conda_environment_name='py_env')
    ```

* **Creating an environment by specifying packages**: using a **CondaDependencies** object:
  
    ```python
    from azureml.core import Environment
    from azureml.core.conda_dependencies import CondaDependencies

    env = Environment('training_environment')
    deps = CondaDependencies.create(conda_packages=['scikit-learn','pandas','numpy'],
                                    pip_packages=['azureml-defaults'])
    env.python.conda_dependencies = deps
    ```

#### Registering and reusing environments

After you've created an environment, you can register it in your workspace and reuse it for future experiments that have the same Python dependencies.

Register it via `env.register(workspace=ws)` and get the registered environments in a workspace using `Environment.list(workspace=ws)`.

#### Retrieving and using an environment

You can retrieve an environment and assign it to an **Estimator** or a **ScriptRunConfig**:

```python
from azureml.core import Environment, Estimator

training_env = Environment.get(workspace=ws, name='training_environment')
estimator = Estimator(source_directory='experiment_folder'
                      entry_script='training_script.py',
                      compute_target='local',
                      environment_definition=training_env)
```

> OBS: When an experiment based on the estimator is run, Azure Machine Learning will look for an existing environment that matches the definition, and if none is found a new environment will be created based on the registered environment specification.

### Introduction to compute targets

Compute Targets are physical or virtual computers on which experiments are run. You can assign experiments to specific compute targets. This means that one can test on cheaper ones and run individual processes on GPUs, if needed.

You pay-by-use as compute targets

* Start on-demand and stop automatically when no longer required.
* Scale automatically based on workload processing needs (for model training)

#### Types of compute

* **Local compute**: Great for test and development. The experiment will run where the code is initiated, e.g., you own computer or a VM with jupyter on top.
* **Training Clusters**: multi-node clusters of VMs that automatically scale up or down to meet demand for training workloads. Useful when working with large data or when needing parallel processing.
* **Inference clusters**: To deploy trained models as production services. They use containerization to enable rapid initialization of compute for on-demand inferencing.
* **Attached compute**: You can attach another Azure-based compute environment to AML, as another VM or a Databricks cluster. They can be used for certain types of workload.

More info [here](https://docs.microsoft.com/en-us/azure/machine-learning/concept-compute-target).

### Create compute targets

Can be done via UI or SDK. UI is the most common.

#### Creating a managed compute target with the SDK

They are managed by AML, e.g., a training cluster.

```python
from azureml.core import Workspace
from azureml.core.compute import ComputeTarget, AmlCompute

# Load the workspace from the saved config file
ws = Workspace.from_config()

# Specify a name for the compute (unique within the workspace)
compute_name = 'aml-cluster'

# Define compute configuration
compute_config = AmlCompute.provisioning_configuration(vm_size='STANDARD_DS12_V2',
                                                       min_nodes=0, max_nodes=4,
                                                       vm_priority='dedicated')

# Create the compute
aml_cluster = ComputeTarget.create(ws, compute_name, compute_config)
aml_cluster.wait_for_completion(show_output=True)
```

> Priority can be **dedicated** to use for this cluster or **low priority**, for less cost but the possibility to be preemted.

#### Attaching an unmanaged compute target with the SDK

Unmanaged instances are defined and managed outside of the AML, e.g., a VM or a Databricks.

```python
from azureml.core import Workspace
from azureml.core.compute import ComputeTarget, DatabricksCompute

# Load the workspace from the saved config file
ws = Workspace.from_config()

# Specify a name for the compute (unique within the workspace)
compute_name = 'db_cluster'

# Define configuration for existing Azure Databricks cluster
db_workspace_name = 'db_workspace'
db_resource_group = 'db_resource_group'
db_access_token = '1234-abc-5678-defg-90...'
db_config = DatabricksCompute.attach_configuration(resource_group=db_resource_group,
                                                   workspace_name=db_workspace_name,
                                                   access_token=db_access_token)

# Create the compute
databricks_compute = ComputeTarget.attach(ws, compute_name, db_config)
databricks_compute.wait_for_completion(True)
```

#### Checking for an existing compute target

You can check if a compute targets exists to only create it otherwise:

```python
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException

compute_name = "aml-cluster"

# Check if the compute target exists
try:
    aml_cluster = ComputeTarget(workspace=ws, name=compute_name)
    print('Found existing cluster.')
except ComputeTargetException:
    # If not, create it
    compute_config = AmlCompute.provisioning_configuration(vm_size='STANDARD_DS12_V2',
                                                           max_nodes=4)
    aml_cluster = ComputeTarget.create(ws, compute_name, compute_config)

aml_cluster.wait_for_completion(show_output=True)
```

More info [here](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-set-up-training-targets).

### Use compute targets

You can use them to run specific workloads:

```python
from azureml.core import Environment, Estimator

compute_name = 'aml-cluster'

training_env = Environment.get(workspace=ws, name='training_environment')

estimator = Estimator(source_directory='experiment_folder',
                      entry_script='training_script.py',
                      environment_definition=training_env,
                      compute_target=compute_name)
```

> OBS: When an experiment for the estimator is submitted, the run will be queued while the compute target is started and the specified environment deployed to it, and then the run will be processed on the compute environment.

Instead of working by name, you could also pass a **ComputeTarget** object:

```python
from azureml.core import Environment, Estimator
from azureml.core.compute import ComputeTarget

compute_name = 'aml-cluster'
training_cluster = ComputeTarget(workspace=ws, name=compute_name)

training_env = Environment.get(workspace=ws, name='training_environment')

estimator = Estimator(source_directory='experiment_folder',
                      entry_script='training_script.py',
                      environment_definition=training_env,
                      compute_target=training_cluster)
```

### Knowledge Check

1. You're using the Azure Machine Learning Python SDK to run experiments. You need to create an environment from a Conda configuration (.yml) file. Which method of the Environment class should you use?

   * create
   * create_from_conda_specification
   * create_from_existing_conda_environment

    <details>
    <summary> 
    Answer
    </summary>
    <p>
     Use the create_from_conda_specification method to create an environment from a configuration file. The create method requires you to explicitly specify conda and pip packages, and the create_from_existing_conda_environment requires an existing environment on the computer.
    </p>
    </details>

1. You must create a compute target for training experiments that require a graphical processing unit (GPU). You want to be able to scale the compute so that multiple nodes are started automatically as required. Which kind of compute target should you create?

   * Compute Instance
   * Training Cluster
   * Inference Cluster

    <details>
    <summary> 
    Answer
    </summary>
    <p>
    Use a training cluster to create multiple nodes of GPU-enabled VMs that are started automatically as needed.
    </p>
    </details>

</p>
</details>

---


## Orchestrating machine learning with pipelines

<details>
<summary> 
Show content
</summary>
<p>

### Learning objectives

* Create an Azure Machine Learning pipeline.
* Publish an Azure Machine Learning pipeline.
* Schedule an Azure Machine Learning pipeline.

### Introduction to pipelines

A pipeline is a workflow of machine learning tasks in which each task is implemented as a step. Steps can be sequential or parallel and you can choose a specific compute target for them to run on.

A pipeline can be executed as a process by running the pipeline as an experiment.

They can be triggered via an scheduler or through a REST endpoint.

#### Pipeline steps

There are different types of steps:
* **PythonScriptStep**: runs a specific python script.
* **EstimatorStep**: runs an estimator.
* **DataTransferStep**: Uses Azure Data Factory to copy data between data stores.
* **DatabricksStep**: runs a notebook, script or compiled JAR on dbks.
* **AdlaStep**: runs a U-SQL job in Azure Data Lake Analytics.

You can find the full list [here](https://docs.microsoft.com/en-us/python/api/azureml-pipeline-steps/azureml.pipeline.steps?view=azure-ml-py).

#### Defining steps in a pipeline

First, you define the steps and then assemble the pipeline based on those:

```python
from azureml.pipeline.steps import PythonScriptStep, EstimatorStep

# Step to run a Python script
step1 = PythonScriptStep(name = 'prepare data',
                         source_directory = 'scripts',
                         script_name = 'data_prep.py',
                         compute_target = 'aml-cluster',
                         runconfig = run_config)

# Step to run an estimator
step2 = EstimatorStep(name = 'train model',
                      estimator = sk_estimator,
                      compute_target = 'aml-cluster')

from azureml.pipeline.core import Pipeline
from azureml.core import Experiment

# Construct the pipeline
train_pipeline = Pipeline(workspace = ws, steps = [step1,step2])

# Create an experiment and run the pipeline
experiment = Experiment(workspace = ws, name = 'training-pipeline')
pipeline_run = experiment.submit(train_pipeline)
```

### Pass data between pipeline steps

It is not unusual to have steps depending on previous steps' results.

#### The PipelineData object

The **PipelineData** object is a special kind of **DataReference** that:

* References a location in a datastore.
* Creates a data dependency between pipeline steps.

It is an intermediary store between two subsequent steps: `step1 -> PipelineData -> step2`.

#### PipelineData step inputs and outputs

To use a **PipelineData** object you must:
1. Define a named **PipelineData** object that references a location in a datastore.
2. Configure the input / output of the steps that use it.
3. Pass the **PipelineData** object as a script parameter in steps that run scripts (and add the `argparse` in those scripts, as we do with usual data refs).

```python
from azureml.pipeline.core import PipelineData
from azureml.pipeline.steps import PythonScriptStep, EstimatorStep

# Get a dataset for the initial data
raw_ds = Dataset.get_by_name(ws, 'raw_dataset')

# Define a PipelineData object to pass data between steps
data_store = ws.get_default_datastore()
prepped_data = PipelineData('prepped',  datastore=data_store)

# Step to run a Python script
step1 = PythonScriptStep(name = 'prepare data',
                         source_directory = 'scripts',
                         script_name = 'data_prep.py',
                         compute_target = 'aml-cluster',
                         runconfig = run_config,
                         # Specify dataset as initial input
                         inputs=[raw_ds.as_named_input('raw_data')],
                         # Specify PipelineData as output
                         outputs=[prepped_data],
                         # Also pass as data reference to script
                         arguments = ['--folder', prepped_data])

# Step to run an estimator
step2 = EstimatorStep(name = 'train model',
                      estimator = sk_estimator,
                      compute_target = 'aml-cluster',
                      # Specify PipelineData as input
                      inputs=[prepped_data],
                      # Pass as data reference to estimator script
                      estimator_entry_script_arguments=['--folder', prepped_data])
```

### Reuse pipeline steps

AML includes some caching and reuse feature to reduce the time to run some steps.

#### Managing step output reuse

By default, the step output from a previous pipeline run is reused without rerunning the step. This is useful if the scripts, sources and directories have no change at all, otherwise this may lead to stale results.

To control reuse for an individual step, you can use `allow_reuse` parameter:

```python
step1 = PythonScriptStep(name = 'prepare data',
                         ...
                         # Disable step reuse
                         allow_reuse = False)
```

#### Forcing all steps to run

You can force all steps to run regardless of individual reuse by setting the `regenerate_outputs` param at submision time:

```python
pipeline_run = experiment.submit(train_pipeline, regenerate_outputs=True)
```

### Publish pipelines

After you have created a pipeline, you can publish it to create a REST endpoint through which the pipeline can be run on demand.

```python
published_pipeline = pipeline.publish(name='training_pipeline',
                                      description='Model training pipeline',
                                      version='1.0')
```

You can also publish the pipeline on a successful run:

```python
# Get the most recent run of the pipeline
pipeline_experiment = ws.experiments.get('training-pipeline')
run = list(pipeline_experiment.get_runs())[0]

# Publish the pipeline from the run
published_pipeline = run.publish_pipeline(name='training_pipeline',
                                          description='Model training pipeline',
                                          version='1.0')
```

To get the endpoint

```python
rest_endpoint = published_pipeline.endpoint
print(rest_endpoint)
```

#### Using a published pipeline

To use the endpoint, you need to get the token from a service principal with permission to run the pipeline.

```python
import requests

response = requests.post(rest_endpoint,
                         headers=auth_header,
                         json={"ExperimentName": "run_training_pipeline"})
run_id = response.json()["Id"]
print(run_id)
```

### Use pipeline parameters

To define parameters for a pipeline, create a **PipelineParameter** object for each parameter, and specify each parameter in at least one step.

```python
from azureml.pipeline.core.graph import PipelineParameter

reg_param = PipelineParameter(name='reg_rate', default_value=0.01)

...

step2 = EstimatorStep(name = 'train model',
                      estimator = sk_estimator,
                      compute_target = 'aml-cluster',
                      inputs=[prepped],
                      estimator_entry_script_arguments=['--folder', prepped,
                                                        '--reg', reg_param])
```

> OBS: You must define parameters for a pipeline before publishing it.

#### Running a pipeline with a parameter

After publishing a pipeline with a parameter, you can specify it in the JSON payload in the REST call:

```python
response = requests.post(rest_endpoint,
                         headers=auth_header,
                         json={"ExperimentName": "run_training_pipeline",
                               "ParameterAssignments": {"reg_rate": 0.1}})
```

### Schedule pipelines

#### Scheduling a pipeline for periodic intervals

To schedule a pipeline to run at periodic intervals, you must define a **ScheduleRecurrance** that determines the run frequency, and use it to create a **Schedule**.

```python
from azureml.pipeline.core import ScheduleRecurrence, Schedule

daily = ScheduleRecurrence(frequency='Day', interval=1)
pipeline_schedule = Schedule.create(ws, name='Daily Training',
                                        description='trains model every day',
                                        pipeline_id=published_pipeline.id,
                                        experiment_name='Training_Pipeline',
                                        # daily schedule
                                        recurrence=daily)
```

#### Triggering a pipeline run on data changes

You can also monitor a specified path on a datastore. This will become a trigger for a new run.

```python
from azureml.core import Datastore
from azureml.pipeline.core import Schedule

training_datastore = Datastore(workspace=ws, name='blob_data')
pipeline_schedule = Schedule.create(ws, name='Reactive Training',
                                    description='trains model on data change',
                                    pipeline_id=published_pipeline_id,
                                    experiment_name='Training_Pipeline',
                                    datastore=training_datastore,
                                    path_on_datastore='data/training')
```

### Knowledge Check

1. You're creating a pipeline that includes two steps. Step 1 preprocesses some data, and step 2 uses the preprocessed data to train a model. What type of object should you use to pass data from step 1 to step 2 and create a dependency between these steps?

   * Datastore
   * PipelineData
   * Data Reference

    <details>
    <summary> 
    Answer
    </summary>
    <p>
    To pass data between steps in a pipeline, use a PipelineData object.
    </p>
    </details>

2. You've published a pipeline that you want to run every week. You plan to use the Schedule.create method to create the schedule. What kind of object must you create first to configure how frequently the pipeline runs?

   * Datastore
   * PipelineParameter
   * ScheduleRecurrance

    <details>
    <summary> 
    Answer
    </summary>
    <p>
    You need a ScheduleRecurrance object to create a schedule that runs at a regular interval.
    </p>
    </details>

</p>
</details>

---

## Deploying machine learning models with Azure Machine Learning

<details>
<summary> 
Show content
</summary>
<p>

### Learning objectives

* Deploy a model as a real-time inferencing service.
* Consume a real-time inferencing service.
* Troubleshoot service deployment

### Deploying a model as a real-time service

You can deploy a model as a real-time web service to several kinds of compute target:
* Local compute
* Azure ML compute instance
* Azure Container Instance (ACI)
* AKS
* Azure Function
* IoT module

AML uses containers for model packaging and deployment.

#### 1. Register a trained model

After a successful training, you first need to register the model.

To register from a local file:

```python
from azureml.core import Model

classification_model = Model.register(workspace=ws,
                       model_name='classification_model',
                       model_path='model.pkl', # local path
                       description='A classification model')
```

Or to reference to the **Run** used to train the model:

```python
run.register_model( model_name='classification_model',
                    model_path='outputs/model.pkl', # run outputs path
                    description='A classification model')
```

#### 2. Define an Inference Configuration

The model will be deployed as a service that consist of:

* A script to load the model and return predictions for submitted data.
* An environment in which the script will be run.

##### Creating an Entry Script (or scoring script)

It is a py file that must contain

* `init()`: Called when the service is initialized.
* `run(raw_data)`: Called when new data is submitted to the service.

```python
import json
import joblib
import numpy as np
from azureml.core.model import Model

# Called when the service is loaded
def init():
    global model
    # Get the path to the registered model file and load it
    model_path = Model.get_model_path('classification_model')
    model = joblib.load(model_path)

# Called when a request is received
def run(raw_data):
    # Get the input data as a numpy array
    data = np.array(json.loads(raw_data)['data'])
    # Get a prediction from the model
    predictions = model.predict(data)
    # Return the predictions as any JSON serializable format
    return predictions.tolist()
```

##### Creating an Environment

You can use **CondaDependencies**

```python
from azureml.core.conda_dependencies import CondaDependencies

# Add the dependencies for your model
myenv = CondaDependencies()
myenv.add_conda_package("scikit-learn")

# Save the environment config as a .yml file
env_file = 'service_files/env.yml'
with open(env_file,"w") as f:
    f.write(myenv.serialize_to_string())
print("Saved dependency info in", env_file)
```

##### Combining the Script and Environment in an InferenceConfig

```python
from azureml.core.model import InferenceConfig

classifier_inference_config = InferenceConfig(runtime= "python",
                                              source_directory = 'service_files',
                                              entry_script="score.py",
                                              conda_file="env.yml")
```

#### 3. Define a Deployment Configuration

Now, select the compute target to deploy to.

> OBS: if deploying to AKS, create the cluster and a compute target for it before deploying.

```python
from azureml.core.compute import ComputeTarget, AksCompute

cluster_name = 'aks-cluster'
compute_config = AksCompute.provisioning_configuration(location='eastus')
production_cluster = ComputeTarget.create(ws, cluster_name, compute_config)
production_cluster.wait_for_completion(show_output=True)
```

With the compute target created, define the deployment config

```python
from azureml.core.webservice import AksWebservice

classifier_deploy_config = AksWebservice.deploy_configuration(cpu_cores = 1,
                                                              memory_gb = 1)
```

The code to configure an ACI deployment is similar, except that you do not need to explicitly create an ACI compute target, and you must use the deploy_configuration class from the **azureml.core.webservice.AciWebservice** namespace. Similarly, you can use the **azureml.core.webservice.LocalWebservice** namespace to configure a local Docker-based service.

#### 4. Deploy the Model

```python
from azureml.core.model import Model

model = ws.models['classification_model']
service = Model.deploy(workspace=ws,
                       name = 'classifier-service',
                       models = [model],
                       inference_config = classifier_inference_config,
                       deployment_config = classifier_deploy_config,
                       deployment_target = production_cluster)
service.wait_for_deployment(show_output = True)
```

For ACI or local services, you can omit the deployment_target parameter (or set it to None).

### Consuming a real-time inferencing service

#### Using the Azure Machine Learning SDK

For testing, you can use the AML SDK

```python
import json

# An array of new data cases
x_new = [[0.1,2.3,4.1,2.0],
         [0.2,1.8,3.9,2.1]]

# Convert the array to a serializable list in a JSON document
json_data = json.dumps({"data": x_new})

# Call the web service, passing the input data
response = service.run(input_data = json_data)

# Get the predictions
predictions = json.loads(response)

# Print the predicted class for each case.
for i in range(len(x_new)):
    print (x_new[i]), predictions[i] )
```

#### Using a REST Endpoint

You can retrieve the service endpoint via the UI or the SDK:

```python
endpoint = service.scoring_uri
print(endpoint)
```

```python
import requests
import json

# An array of new data cases
x_new = [[0.1,2.3,4.1,2.0],
         [0.2,1.8,3.9,2.1]]

# Convert the array to a serializable list in a JSON document
json_data = json.dumps({"data": x_new})

# Set the content type in the request headers
request_headers = { 'Content-Type':'application/json' }

# Call the service
response = requests.post(url = endpoint,
                         data = json_data,
                         headers = request_headers)

# Get the predictions from the JSON response
predictions = json.loads(response.json())

# Print the predicted class for each case.
for i in range(len(x_new)):
    print (x_new[i]), predictions[i] )
```

#### Authentication

There are two kinds of auth

* **Key**: Requests are authenticated by specifying the key associated with the service.
* **Token**: Requests are authenticated by providing a JSON Web Token (JWT).

> OBS: By default, authentication is disabled for ACI services, and set to key-based authentication for AKS services (for which primary and secondary keys are automatically generated). You can optionally configure an AKS service to use token-based authentication (which is not supported for ACI services).

You can retrieve the keys for a **WebService** as

```python
primary_key, secondary_key = service.get_keys()
```

To use a token, the application needs to use a service-principal auth to verity the identity through AAD and call the **get_token** method to create a time-limited token.

```python
import requests
import json

# An array of new data cases
x_new = [[0.1,2.3,4.1,2.0],
         [0.2,1.8,3.9,2.1]]

# Convert the array to a serializable list in a JSON document
json_data = json.dumps({"data": x_new})

# Set the content type in the request headers
request_headers = { "Content-Type":"application/json",
                    "Authorization":"Bearer " + key_or_token }

# Call the service
response = requests.post(url = endpoint,
                         data = json_data,
                         headers = request_headers)

# Get the predictions from the JSON response
predictions = json.loads(response.json())

# Print the predicted class for each case.
for i in range(len(x_new)):
    print (x_new[i]), predictions[i] )
```

### Troubleshooting service deployment

#### Check the Service State

```python
from azureml.core.webservice import AksWebservice

# Get the deployed service
service = AciWebservice(name='classifier-service', workspace=ws)

# Check its state
print(service.state)
```

> OBS: To view the state of a service, you must use the compute-specific service type (for example AksWebservice) and not a generic WebService object.

#### Review Service Logs

```python
print(service.get_logs())
```

#### Deploy to a Local Container

A quick check on runtime errors can be done by deploying to a local container.

```python
from azureml.core.webservice import LocalWebservice

deployment_config = LocalWebservice.deploy_configuration(port=8890)
service = Model.deploy(ws, 'test-svc', [model], inference_config, deployment_config)
```

You can then test the locally deployed service using the SDK `service.run(input_data = json_data)` and troubleshoot runtime issues by making changes to the scoring file and reloading the service without redeploying (this can ONLY be done with a local service)

```python
service.reload()
print(service.run(input_data = json_data))
```

### Check your knowledge

1. You've trained a model using the Python SDK for Azure Machine Learning. You want to deploy the model as a containerized real-time service with high scalability and security. What kind of compute should you create to host the service?

    * An Azure Kubernetes Services (AKS) inferencing cluster.
    * A compute instance with GPUs.
    * A training cluster with multiple nodes.

    <details>
    <summary> 
    Answer
    </summary>
    <p>
    You should use an AKS cluster to deploy a model as a scalable, secure, containerized service.
    </p>
    </details>

2. You're deploying a model as a real-time inferencing service. What functions must the entry script for the service include?

    * main() and score()
    * base() and train()
    * init() and run()

    <details>
    <summary> 
    Answer
    </summary>
    <p>
    You must implement init and run functions in the entry (scoring) script.
    </p>
    </details>

</p>
</details>

---

## Automate machine learning model selection with Azure Machine Learning

<details>
<summary> 
Show content
</summary>
<p>

### Learning objectives
