# Titanic example from Kaggle
## Project Workflow:

### Step 1 : Create a Project in DKube
1. Click on Projects in left pane in Dkube.
2. Click on + Create Project.
3. Give a project name, say titanic-{user}, replace {user} with your username.

### Step 2 :Upload Train & Eval dataset: 
1. Click on Repos in left pane and then click on +dataset.
2. Details to be filled for train dataset:
   - Name: titanic-train-ds
   - DataSource: Other 
   - URL: https://dkube.s3.amazonaws.com/datasets/titanic/train.csv
3. Details to be filled for test dataset
   - Name: titanic-test-ds
   - DataSource: Other
   - URL: https://dkube.s3.amazonaws.com/datasets/titanic/test.csv

## Data Scientist Workflow 

### Step 3: Create dkube code repo:
1. Click on Repos in left pane and then click on +Code.
2. Name: titanic-code-user
3. Git URL: https://github.com/pallavi-pannu-oc/titanic-user.git
4. Branch: main

### Step 4: Create a model 
1. Click on Repos in left pane and then click on Create model, 
2. Name: titanic-model-user

### Step 5 : Create Featuresets
1. Click on Featuresets in left pane.
2. Click on +Featureset and give a name.
   - titanic-train-fs-{user}, replace {user} with your username.
   - Spec upload:none
3. Similarly create a test featureset.
   - titanic-test-fs-{user}, replace {user} with your username.
   - Spec upload:none

### Step 6 : Launch JupyterLab IDE
1. Click on IDEs in left pane and then select your titanic project from top.
2. Click on +JupyterLab and then fill the below details:
   - Give a name : titanic-{user}, replace {user} with your username.
   - Select code as titanic-code-user.
   - Select Framework as tensorflow and version as 2.0.0
3. Go to workspace/titanic-code-user and then run all the cells of pipeline.ipynb file.

