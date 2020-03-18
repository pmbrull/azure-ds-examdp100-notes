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