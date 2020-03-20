# Get started with Machine Learning with an Azure Data Science Virtual Machine

## Introduction to the Azure Data Science Virtual Machine (DSVM)

<details>
<summary> 
Show content
</summary>
<p>

### Learning objectives

* Learn about the types of Data Science Virtual Machines
* Learn what type of DSVM to use for each type of use case

### When to use an Azure DSVM?

Azure DSVM makes it easy to maintain consistency in the evolving Data Science environments.

It also provides samples in Jupyter Notebooks and scripts for Python and R to learn about Microsoft and Azure ML services:
* How to connect to cloud Datastores with Azure ML and how to build models.
* Deep Learning samples using Microsoft Cognitive Services.
* How to compare Microsoft R and open source R and how to operationalize models with ML Services in SQL Server.

### Types of Azure DSVM

* **Windows vs. Linux**: Windows Server 2012 and 2016 vs. Ubuntu 16.04 LTS and CentOS 7.4
* **Deep Learning**: The Deep Learning DSVM comes preconfigured and preinstalled with many tools and you can select high-speed GPU based machines.
* **Geo AI DSVM**: VM optimized for geospatial and location data. It has ArcGIS Pro system integrated. 

### Use cases for a DSVM

* **Collaborate as a team using DSVMs**: Working with cloud-based resources that can share the same configuration helps to ensure that all team members have a consistent development environment.
* **Address issues with DSVMs**: As issues related to environment mismatches are reduced. Giving DSVMs to students in a class.
* **Use on-demand elastic capacity for large-scale projects**: As it helps to replicate data science environments on demand to allow high-powered computing resources to be run.
* **Experiment and evaluate on a DSVM**: As they are easy to create, they can be used for demos and short experiments.
* **Learn about DSVMs and deep learning**: The flexibility of the underlying compute power (scaling or switching to GPU) makes it easy to train all kind of models.

### Knowledge Check

1. Which of the following is a reason to use an Azure Data Science Virtual Machine?

    * You want to create an Azure Databricks workspace.
    * You want to get a jump-start on data science work.
    * You want to deploy a web application to it.

    <details>
    <summary> 
    Answer
    </summary>
    <p>
    The purpose of Data Science Virtual Machines is to give a data scientist the tools they need, pre-installed, and ready to go.
    </p>
    </details>

1. Which of the following is installed on a Data Science Virtual Machine?

    * Azure Data Warehouse
    * Jupyter Notebook
    * Azure Machine Learning Studio

    <details>
    <summary> 
    Answer
    </summary>
    <p>
     Jupyter Notebook is installed on Data Science Virtual Machines and provides a great data science development tool.
    </p>
    </details>

</p>
</details>

---

## Explore the types of Azure Data Science Virtual Machines

<details>
<summary> 
Show content
</summary>
<p>

### Learning objectives

* Learn how to create Windows-based and Linux-based DSVMs
* Explore the Deep Learning Data Science Virtual Machines
* Work with Geo AI Data Science Virtual Machines

### Windows-Based DSVMs

You can use the Windows-based DSVM to jump-start your data science projects. You don't pay for the DSVM image, just usage fees.

The image comes with a bunch of features:
* Tutorials
* Support for Office
* SQL Server integrated with ML Services
* Preinstalled languages: R, Python, SQL, C#
* Data Science tools such as Azure ML SDK for Python, Anaconda, Jupyter...
* ML tools as Azure Congitive Services support, H2O, Tensorflow, Weka...

### Deep Learning Virtual Machine

Deep Learning Virtual Machines (DLVMs) use GPU-based hardware that provide increased mathematical calculation speed for faster model training. The image can be either Windows or Ubuntu.

The DLVM simplifies the tool selection process by including preconfigured tools for different situations.

### Geo AI Data Science VM with ArcGIS

Both Python and R work with ArcGIS Pro, and are preconfigured on the Geo AI Data Science VM.

The image includes a large set of tools as DL frameworks, Keras, Caffe2 and Spark standalone.

> OBS: Tools need to be compatible with GPUs.

It also comes bundled with IDEs such as visual studio or PyCharm.

Examples of Geo AI include:

* Real-time results of traffic conditions
* Driver availability in Uber or Lyft at any time
* Deep learning for disaster response
* Urban growth prediction

### Knowledge Check

1. You want to learn about how to use Azure services related to machine learning with as little fuss as possible installing and configuring software and locating demonstration scripts. Which Data Science Virtual Machine type would best suit these needs?

    * Deep Learning DSVM
    * Windows 2016 DSVM
    * Geo AI Data Science VM with ArcGIS DSVM

    <details>
    <summary> 
    Answer
    </summary>
    <p>
    The Windows 2016 gives you the most popular data science tools installed and configured and includes many sample scripts for using Azure machine learning related services.
    </p>
    </details>

2. You need to train deep learning models to do image recognition using a lot of training data in the form of images. Which DSVM configuration would be best for the fastest model training?

    * Windows 2016 with standard CPUs.
    * Geo AI Data Science VM with ArcGIS DSVM
    * Deep Learning VM which is configured to use GPUs.

    <details>
    <summary> 
    Answer
    </summary>
    <p>
    The DSVM includes all the software needed for training deep learning models and use graphic processor units (GPUs) which perform calculations much faster than standard CPUs.
    </p>
    </details>

</p>
</details>

---

## Provision and use an Azure Data Science Virtual Machine

This module is based on exercise, so it's best followed [here](https://docs.microsoft.com/en-us/learn/modules/provision-and-use-azure-dsvm/).

### Knowledge Check

1. What method did we use to log into a Windows-Based Data Science VM?

    * Remote Desktop Protocol (RDP)
    * HTTP
    * ODBC

    <details>
    <summary> 
    Answer
    </summary>
    <p>
    RDP: A step by step walk through explains all the steps to connect to a Windows-based DSVM.
    </p>
    </details>

1. What development environment has pre-loaded sample code available?

    * PyCharm
    * Zeppelin Notebook
    * Jupyter Notebook

    <details>
    <summary> 
    Answer
    </summary>
    <p>
    Jupyter: We showed that many sample notebooks are installed that demonstrate how to use Microsoft Machine Learning technologies.
    </p>
    </details>

1. What type of Jupyter Notebook cell is used to provide annotations?

    * Code cell
    * Markdown cell
    * Raw cell

    <details>
    <summary> 
    Answer
    </summary>
    <p>
    Markdown support rich formatting and is ideal for adding comments and annotations to your notebooks.
    </p>
    </details>

</p>
</details>

---