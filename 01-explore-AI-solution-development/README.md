# Explore AI solution development with data science services in Azure

## Introduction to Data Science in Azure

<details>
<summary> 
Show content
</summary>
<p>

### Learning Objectives

* Learn the steps involved in the data science process
* Learn the machine learning modeling cycle
* Learn data cleansing and preparation
* Learn model feature engineering
* Learn model training and evaluation
* Learn about model deployment
* Discover the specialized roles in the data science process

### The Data Science process

![img](../assets/img/ds-process.png)

Iterative process that starts with a question, risen from business needs and understanding.

### What is modeling?

Modeling is a cycle of data and business understanding. You start by exploring your assets, in this case data, with **Exploratory Data Analysis (EDA)**, from that point feature engineering starts and finally train a model on top, which is an algorithm that learns information and provides a probabilistic prediction.

In the end, the model is evaluated to check where it is accurate and where it is failing to correct the behavior.

### Choose a use case

Identify the problem (business understanding) -> Define the project goals -> Identify data sources

### Data preparation

Data cleansing and EDA are vital to the modeling process, to get insights on what data is or is not useful and what needs to be corrected or taken into account. Understanding the data is one of the most vital steps in the data science cycle.

### Feature engineering

What extra knowledge we can extract by combining existing features to create new ones.

### Model training

Split data -> Cross-validate data -> Obtain probabilistic prediction

### Model evaluation

**Hyperparameters** are parameters used in model training that cannot be learned by the training process. These parameters must be set before model training begins.

For evaluating the results you need to set up a metric to compare different runs, such as accuracy or MSE.

### Model deployment

Model deployment is the final stage of the data science procedure. It is often done by a developer, and is usually not part of the data scientist role.

### Specialized roles in the Data Science process

In the data science process, there are specialists in each of the steps:

Business Analyst or Domain Expert, Data Engineer, Developer and Data Scientist.

### Knowledge Check

1. Which of the following is not a specialized role in the Data Science Process?

* Database Administrator
* Data Scientist
* Data Engineer

    <details>
    <summary> 
    Answer
    </summary>
    <p>
    DBA
    </p>
    </details>

1. Model feature engineering refers to which of the following?

* Selecting the best model to use for the experiment.
* Determine which data elements will help in making a prediction and preparing these columns to be used in model training.
* Exploring the data to understand it better.

    <details>
    <summary> 
    Answer
    </summary>
    <p>
    Feature engineering involves the data scientist determining which data to use in model training and preparing the data so it can be used by the model.
    </p>
    </details>

1. The Model deployment involves.

* Calling a model to score new data.
* Training a model.
* Copying a trained model and its code dependencies to an environment where it will be used to score new data.

    <details>
    <summary> 
    Answer
    </summary>
    <p>
    Deploying a model makes it available for use.
    </p>
    </details>

</p>
</details>

---

## Choose the Data Science service in Azure you need

<details>
<summary> 
Show content
</summary>
<p>

### Learning Objectives

* Differentiate each of the Azure machine learning products.
* Identify key features of each product.
* Describe the use cases for each service.

### Machine Learning options on Azure

We have the following services:

* **Azure Machine Learning Studio**: GUI-based solution best chosen for learning. It includes all DS pipeline steps, from importing and playing around with data to different deployment options. All is based in a drag-and-drop method.
* **Azure Databricks**: Great collaboration platform with a powerful notebook interface, job scheduling, AAD integration and granular security control. It allows to create and modify Spark clusters.
* **Azure Data Science Virtual Machine**: preconfigured VMs with lots of preinstalled popular ML tools. You can directly connect to the machine via ssh or remote desktop. There are different types of machines:
    * Linux and Windows OS, where Windows supports scalability with ML in SQL Server and Linux does not.
    * Deep Learning VM, offering DL tools.
    * Geo AI DSVM, with specific tools for working with spatial data. Includes ArcGIS.
* **SQL Server Machine Learning Services**: add-on which runs on the SQL Server on-premises and supports scale up and high performance of Python and R code. It includes several advantages:
    * Security, as the processing occurs closer to the data source.
    * Performance
    * Consistency
    * Efficiency, as you can use integrated tools such as PowerBI to report and analyze results.
* **Spark on HDInsight**: HDInsight is PaaS service offering Apache Hadoop. It provides several benefits:
    * Easy and fast to create and modify clusters on demand.
    * Usage of ETL tools in the cluster with Map Reduce and Spark.
    * Compliance standards with Azure Virtual Network, envryption and integration of Azure AD.
    * Integrated with other Azure services, such as ADLS or ADF.

    HDInsight Spark is an implementation of Apache Spark on Azure HDInsight.
* **Azure Machine Learning Service**: Supports the whole DS pipeline integration, scale processing and automate the following tasks:

    * Model management
    * Model training
    * Model selection
    * Hyper-parameter tuning
    * Feature selection
    * Model evaluation

    It supports open-source technologies such as Python and common ds tools. It makes it easier to containerize and deploy the model and automate several tasks. The platform is designed to support three roles:

    * Data Engineer to ingest and prepare data for analysis either locally or on Azure containers.
    * Data Scientist to apply the modeling tools and processes. AMLS support sklearn, tensorFlow, pyTorch, Microsoft Cognitive Toolkit and Apache MXNet.
    * Developer to create an image of the built and trained model with all the needed components. An **image** contains:
        1. The model
        1. A scoring script or application which passes input to the model and returns the output of the model
        1. The required dependencies, such as Python scripts or packages needed by the model or scoring script.

        Images can be deployed as Docker images or field programmable gate array (FPGA) images. Iages can be deployed to a web service (running in Azure Container Instance, FPGA or Azure Kubernetes Service), or an IoT module (IoT Edge).

        > OBS: Scalability is enabled during training, but once the code is deployed it is flat. Also, it is only supported as an Azure App Service so you keep paying even if it is idle.


### Knowledge Check

1. Azure Machine Learning service supports which programming language.

* R
* Julia
* Python

    <details>
    <summary> 
    Answer
    </summary>
    <p>
    Python is supported by Azure Machine Learning service.
    </p>
    </details>

1. Azure Databricks is built on which Big Data platform?

* Azure SQL Data Warehouse
* SQL Server
* Apache Spark

    <details>
    <summary> 
    Answer
    </summary>
    <p>
    Azure Databricks makes using Spark easier.
    </p>
    </details>


1. Which is not an operating system available for an Azure Data Science Virtual Machine?

* Windows
* Linux
* Apple iOS

    <details>
    <summary> 
    Answer
    </summary>
    <p>
    Data Science VMs running Apple iOS are not available.
    </p>
    </details>


</p>
</details>

---