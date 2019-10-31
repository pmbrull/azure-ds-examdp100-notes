# DS DP-100 Exam training 01

## Azure Data Science Options

* Azure ML Studio -> Drag and drop. Understand for the exam. No code is needed there. Training and Deployment. Complete ML environment. Ideal for learning and beginner data scientists.
* Azure Databricks for Big Data - based on Spark. Massive scale with spark. User friendly portal. Dynamic scale. Secure collaboration (secured workspace). DS tools. You can use different languages in the same notebook.
	* Core artifacts: Jobs, libraries, clusters, workspaces and notebooks.
* Azure Data Science Virtual Machine - VM with almost all of the tools one would need to do DS already presintalled. You can deploy them directly to Azure and work from there. It's easy to customize for your needs. It has some sample code already there. They merged it with Deep Learning VM. There are specific versions for Geo data.
* SQL Server Machine Learning Services - We cna use this to analyze data on SQL Server. Useful for on-premise data. It is an option as source Python and R does not scale, security concerns, operationalization.
* Spark on Azure HDInsight - massive scale with in-memory processing. Hortonworks Distribution. Easy management PaaS. Integration with other Azure services.

> Databricks vs. HDInsights: Databricks it's easier to collaborate, built for collaboration and work in teams.

* Azure ML Service: core of the course. Model management, training, selection, hyper-param tuning, feature selection and model evaluation. It lets you automate tuning and selection tasks. All is in Python.

## Azure Notebooks
Azure based Jupyter Notebooks. Free tier. Ready to use project that teach how to use Azure data and AI services.

Jupyter notebooks can be integrated in VScode and thus you can use git integration with that.

Azure notebooks only support Python, R and F#.

The advantadge of Azure Notebooks is that most of the libraries are preinstalled, but you still have the possibility to install more libraries. You can upload data from your local machine and use custom environment configuration. By using VMs from your azure subscription you can add processing power.

It is Azure ML service ready. From Azure Notebooks you can call Azure ML Service.

## Azure ML Service

Bring the power of containerization and automation to DS. Pack model and libraries into a container and run everything.

DS pipeline:
	Environment setup -> Data preparation -> Experimentation -> Deployment

* Environment setup: create a workspace to store your work. Use python or Azure portal. An experiment within the workspace stored model training information. Use the IDE of your choice.
* Data Preparation: Use python libraries or the Azure Data Prep SDK.
* Experimentation: Train models with Python open source modules of your choice. Train locally or in Azure. Submit model training to Azure containers. Monitor model training. Register final model.
* Deployment: to make a model available. Target deployment environments are: Docker images, Azure container instances, Azure kubernetes service, Azure IoT edge, Field Programmable Gate Array (FPGA). FOr the dpeloyment you'll need the following files:
	* A score script file tells Azure ML Services (AMLS) to call the model
	* An environment file specifies package dependencies
	* A configuration file requests the required resources for the container.

### What is a Workspace

The top-level resource for AMLS. It serves as a hub for building and deploying models. You can create a workspace in the Azure portla, or you can create and access it using Python on an IDE of your choice.

All models must be registered in the workspace for future use. Together with the scoring scripts, you create an image for deployment.

The workspace stores experiment objects that are required for each model you create. Additionally, it saves your compute targets. You can track training runs.

### What is an Image

An image has three components:
* A model and scoring script or application
* An environment file that declares the dependencies that are needed by the model, scoring script or application
* A configuration file that describes the necessary resources to execute the model

### What is a Datastore

An abstraction over an Azure Storage account. Each workspace has a registered, default datastore that you can use right away, but you can register other Azure Blob or File storage containers as a datastore.

### What is a Pipeline

A ML pipeline is a tool to create and manage workflows during a DS process: data manipulation, model training and testing and deployment phases. Each step of the process can run unattended in different compute targets, which makes it easier to allocate resources.

### What is a Compute Target

Is the compute resource to run a training script or to host service deployment. It's attached to a workspace. Other than the local machine, users of the workspace share compute targets.

### What is a deployed Web Service

For a deployed web service, you have the choices of container Instances, AKS or FPGAs. With your model, script and associated files all set in the image, you can create a web service.

> OBS: Trainer said that its better to create a new RG for each ML workspace as there are several resources involved and we don't want to get a mess.

> OBS2: In Azure Notebooks, change python kernel to 3.6, as the default is just set to Python 3! This can rise errors when importing azure ML libs.

### Interact with ML Service

We can interact via Azure Notebook (linking the subscription), a visual interface, Notebook VMs and automated ML.

You can run notebooks in the workspace without any kind of authentication and they are stored in the WS. So it is useful to work in teams.

If you use JupyterLab, you can link a repo in Azure DevOps.

If you register a model with the same name multiple times, it gets uploaded with greater version.

> OBS: Scalability is enabled during training, but once the code is deployed it is flat. Also, it is only supported as an Azure App Service so you keep paying even if it is idle.
