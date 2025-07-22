Employee Salary Prediction
This project, part of the IBM SkillsBuild program, is a complete data science pipeline for predicting employee salaries. It uses various classification algorithms to determine if an individual's income is above or below $50,000 based on demographic and employment data. The project encompasses everything from data cleaning and preprocessing to model training, evaluation, and deployment as an interactive web application.


>> Problem Statement
Determining fair and accurate employee salaries is a significant challenge for organizations. Salaries are influenced by a multitude of factors including age, education, occupation, and work hours. This project aims to analyze employee data to identify key attributes influencing earnings and to build a predictive model that can bring transparency and equity to compensation structures.



✨ Features

Real-Time Prediction: An interactive interface to predict an individual's salary class based on input features.


Batch Prediction: Allows users to upload a CSV file with multiple employee records and receive salary predictions for all of them.


User-Friendly Interface: Built with Streamlit to be accessible for both technical and non-technical users.

>> Project Workflow
The project follows a systematic machine learning workflow:


Data Collection: The UCI Adult Income Dataset is used for this project.

Data Preprocessing:


Handling Missing Values: Missing data, represented by '?', were replaced with the "Others" category.


Removing Irrelevant Data: Rows with 'Without-pay' and 'Never-worked' in the workclass column were removed to focus on the relevant workforce.


Outlier Removal: Box plots were used to identify and filter extreme values in features like age and hours-per-week.



Feature Transformation: The 'education' column was dropped due to its high correlation with 'educational-num'. Categorical features were converted into numerical format using Label Encoding.


Model Training and Evaluation:

The data was split into 80% for training and 20% for testing.

Five different classification models were trained and evaluated: Logistic Regression, Random Forest, K-Nearest Neighbors (KNN), Support Vector Machine (SVM), and Gradient Boosting Classifier.

A 

Pipeline was used for each model to include StandardScaler for feature scaling.

Model Selection & Deployment:

The Gradient Boosting model was selected as the best-performing model based on accuracy and other metrics.

The trained model was saved using 

joblib for deployment.

A web application was developed using Streamlit to serve the model.

⚙️ Technologies & Libraries Used

System: Windows 10 / Linux / macOS 

Core Libraries:

Python: The core programming language.


Pandas: For data manipulation and analysis.


Scikit-learn: For building and training machine learning models.


NumPy: For numerical operations.

Web Framework:


Streamlit: To create the interactive web application.

Model Management:


Joblib: For saving and loading the trained model.

>> Setup and Usage
To run this project locally, follow these steps:

Clone the repository:

Bash

git clone [https://github.com/code2104-sys/Employee_Salary_Predictor.git]
cd Employee_Salary_Predictor
Install the required libraries:
Ensure you have at least 4 GB of RAM and 500 MB of free storage space.


Bash

pip install pandas scikit-learn streamlit numpy joblib
Run the Streamlit application:

Bash

streamlit run app.py
📊 Results
The performance of the five machine learning models was evaluated, and the Gradient Boosting classifier achieved the highest accuracy.

Model	Accuracy
Logistic Regression	
0.8151 

Random Forest	
0.8508 

K-Nearest Neighbors (KNN)	
0.8243 

Support Vector Machine (SVM)	N/A
Gradient Boosting		
0.8557 


Export to Sheets
Based on these results, the 

Gradient Boosting model was selected for deployment in the final web application.

💡 Conclusion
This project successfully demonstrates an end-to-end machine learning workflow, from initial data cleaning to deploying a user-friendly web application. The resulting system provides a data-driven approach that can assist in making more informed and equitable decisions regarding employee compensation.


🔭 Future Scope

Real-Time Data Integration: Connect with live HR management systems to continuously learn from new data.


Advanced Model Enhancements: Utilize techniques like Grid Search and Cross-Validation to further optimize model performance.


Expanded Feature Set: Incorporate additional features like job location, industry sector, and company size for more robust predictions.


Cloud Deployment: Deploy the application on cloud platforms like AWS, Azure, or Google Cloud for better scalability and accessibility.

>> Presented By
Name: Namneet Dash

College: Trident Academy of Technology

Department: Computer Science and Engineering


AICTE Internship Student Regd ID: STU665ae065302cc1717231717
