### Introduction 

This repository contains the code for the development and deployment of a basic NN for the Iris flowers classification problem using Tensorflow, Flask and Docker.

📊 Dataset:
The dataset contains measurements of Iris flowers' sepal length, sepal width, petal length, and petal width. The goal is to build a model that can accurately predict the species of the Iris flower based on these features.

🚀 Deployment with Flask:
I leverage Flask, a powerful and lightweight Python web framework, to deploy my trained Iris flower classification model as a web application. Flask allows to create a user-friendly API, enabling users to make real-time predictions by submitting measurements through a very simple interface.

🐳 Docker for Containerization:
To ensure seamless deployment and scalability, I containerize the Flask application using Docker. Docker enables to package the Flask app and its dependencies into a single container, ensuring consistent behavior across different environments. This makes it easier for users to run the application without worrying about complex setup procedures.

### Instructions 

Firstly, make sure you have installed Docker.

1. Create an environment of your preference, but make sure it includes tensorflow, flask and scikit-learn.

2. A model checkpoint is provided but in case you want to retrain it you can do it by doing:

`python train.py`

3. Create a REST API with Flask. You can try several things, but make sure you run the script first:

`python app.py` 

and then access localhost on your browser.

4. Create a Docker Image and Run Docker

For this purpose you are going to use the Dockerfile.

`docker build -t iris_demo_v1 .`

After the image is created you can run:

`docker run -p 5000:3000 iris_demo_v1`

5. Lastly, you can push this code to the Docker repository which can be used to run the app.




