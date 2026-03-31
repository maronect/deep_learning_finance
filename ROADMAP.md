Prompt — Roadmap to transform DeepLearning Finance into an ML Engineering Project

The project must evolve from an academic Machine Learning study focused on financial portfolio optimization into a structured Machine Learning Engineering application, emphasizing reproducibility, deployment, pipeline automation, and model serving through an API. The system must be capable of collecting financial market data, processing and generating features, training Machine Learning models to estimate expected asset returns, performing portfolio optimization based on the Markowitz framework, and making results available through an API, while supporting automatic periodic updates. The development process should follow structured stages, prioritizing modular architecture, maintainability, and clarity of design.

## Stage 1 — Project structure refactoring and modularization

Objective: transform the current research-oriented code into a structured, modular, and reusable application.

The repository must be reorganized to clearly separate responsibilities among components of the system. The architecture should support maintainability, scalability, and reuse of modules related to data processing, modeling, and optimization.

The structure should include dedicated directories for data ingestion, feature engineering, machine learning models, portfolio optimization logic, configuration files, and shared utilities. A directory for storing artifacts must also be included, allowing storage of model outputs such as predictions, metrics, and portfolio weights.

The pipeline should be logically separated into independent steps:

financial data collection from market APIs
calculation of returns and statistical metrics
feature generation for supervised learning models
training of Machine Learning models
generation of expected return predictions
portfolio optimization using the Markowitz framework
evaluation and comparison of portfolio strategies

The code must allow both full pipeline execution and partial execution of specific stages.

## Stage 2 — Standardization of data and training pipeline

Objective: ensure reproducibility and consistency of the training process.

A standardized pipeline must be created to execute all stages of data preparation, training, and prediction deterministically. The pipeline must allow parameterization of relevant variables such as:

selected financial assets
data frequency (daily, weekly, monthly)
training time window
rolling window sizes used for feature generation
Machine Learning model choice
risk-free rate used in Sharpe ratio calculation

The pipeline must generate and persist relevant artifacts including:

processed historical returns
generated feature datasets
trained model parameters
predicted expected returns
optimized portfolio weights
evaluation metrics for models
portfolio performance metrics

Artifacts must be stored in a structured and organized way to allow experiment traceability and reproducibility.

## Stage 3 — API implementation using FastAPI

Objective: transform the project into an accessible service.

An API must be developed using FastAPI in order to expose the model outputs and enable interaction with the system. The API must provide endpoints that allow retrieval of predictions, execution of the training pipeline, and access to portfolio optimization results.

The API must include endpoints for:

application health check
retrieving available assets
retrieving predicted expected returns
triggering model training
computing optimal portfolio allocation
retrieving efficient frontier data
retrieving model performance metrics
retrieving portfolio performance metrics

The API must use Pydantic schemas to validate request inputs and ensure consistency of responses.

Automatic API documentation must be available via Swagger.

## Stage 4 — Persistence of results and model artifacts

Objective: ensure traceability and historical record of pipeline executions.

Pipeline outputs must be persistently stored to allow comparison between different executions. Stored information should include:

execution timestamp
assets used in the optimization process
predicted expected returns
optimized portfolio weights
model performance metrics
portfolio risk and return metrics

Persistence can initially be implemented using structured files or a relational database.

The system must allow retrieval of past execution results for comparison and evaluation.

## Stage 5 — Containerization with Docker

Objective: ensure reproducibility of the environment and simplify deployment.

The application must be containerized using Docker to ensure consistent execution across different environments.

The container must include:

Python dependencies
application source code
API configuration
pipeline execution scripts

The system must be executable locally using a single command.

Docker Compose may be used to orchestrate multiple services if necessary, such as API services and databases.

## Stage 6 — Automated testing

Objective: ensure reliability and correctness of the system.

Automated tests must be implemented to validate:

return calculation logic
volatility calculation logic
portfolio optimization functionality
feature generation logic
API endpoint behavior
prediction consistency

Tests must be automatically executed when code changes occur.

## Stage 7 — CI/CD configuration

Objective: automate validation and deployment processes.

A continuous integration pipeline must be configured to automatically perform:

dependency installation
automated test execution
code validation
Docker image build

The CI/CD pipeline must ensure that only validated versions of the project are deployed.

## Stage 8 — Automated data update and model retraining

Objective: transform the system into a continuously operating service.

The application must include a mechanism for periodic execution of the pipeline, including:

retrieval of updated financial data
feature recalculation
generation of updated expected return predictions
recalculation of optimal portfolio allocation
updating stored performance metrics

Updates must occur automatically at defined intervals without manual intervention.

Execution logs must be generated to support monitoring and debugging.

## Stage 9 — Application deployment

Objective: make the application accessible online.

The application must be deployed in an environment accessible via the internet, allowing external access to the API endpoints.

Deployment configuration must consider:

environment variables
application stability
API accessibility
availability of documentation endpoints
## Stage 10 — Project documentation

Objective: ensure clarity and usability of the project.

The documentation must clearly describe:

project objectives
system architecture
data pipeline workflow
Machine Learning methodology
portfolio optimization logic
local execution instructions
deployment instructions
API endpoint descriptions
technology stack used

The documentation must highlight the Machine Learning Engineering and MLOps aspects of the project, demonstrating practical integration between Machine Learning and software engineering practices applied to quantitative finance.

Expected outcome

By the end of these stages, the project should become a complete Machine Learning Engineering application capable of:

automatically collecting financial market data
generating return predictions using Machine Learning models
optimizing portfolios using predicted expected returns
exposing results through an API
periodically updating predictions and portfolio allocations
maintaining historical execution records
running in a containerized environment
supporting automated testing and deployment pipelines

The project should demonstrate practical skills in Machine Learning, software engineering, API development, containerization, CI/CD automation, reproducibility, and applied artificial intelligence within the context of quantitative finance.