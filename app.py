# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
# from flask import Flask, request, jsonify
# from flask_cors import CORS  # Import CORS from flask_cors
# import requests
# from bs4 import BeautifulSoup
# from selenium import webdriver
# from selenium.webdriver.chrome.service import Service
# from selenium.webdriver.common.by import By
# from selenium.webdriver.support.ui import WebDriverWait
# from selenium.webdriver.support import expected_conditions as EC
# from flask import Flask, request, jsonify
# from flask_cors import CORS


# app = Flask(__name__)
# CORS(app)  # Enable CORS for all routes

# # Your training data, model creation, and classifier code should be here
# training_data = pd.read_csv('./datasets/Training.csv')
# testing_data = pd.read_csv('./datasets/Testing.csv')
# training_data = training_data[training_data.columns[:-1]]
# all_symptoms = training_data.columns
# final_symptoms  = []
# for i in all_symptoms:
#     final_symptoms.append(i.replace('_', ' '))
# training_data.columns = final_symptoms
# testing_data.columns = final_symptoms
# final_symptoms = final_symptoms[:-1]
# output_file = open('all_symptoms.txt', 'w')

# for symptom in final_symptoms:
#     output_file.write('"' + symptom + '", ')

# output_file.close()


# # Split the dataset into features (X) and target (y)
# X_train = training_data.iloc[:, :132]  # Features
# y_train = training_data.iloc[:, -1]   # Target
# X_test = testing_data.iloc[:, :132]  # Features
# y_test = testing_data.iloc[:, -1]   # Target

# # Create and train the Naive Bayes classifier
# naive_bayes_classifier = MultinomialNB()
# naive_bayes_classifier.fit(X_train, y_train)

# # Create and train the Decision Tree classifier
# decision_tree_classifier = DecisionTreeClassifier(random_state=42)
# decision_tree_classifier.fit(X_train, y_train)

# # Create and train the Random Forest classifier
# random_forest_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
# random_forest_classifier.fit(X_train, y_train)

# # Define endpoint for disease prediction
# @app.route('/predict_disease', methods=['POST'])
# def predict_disease():
#     data = request.json  # Get JSON data from frontend
#     symptoms = data.get('symptoms', [])  # Get symptoms array from JSON
    
#     user_input = [0] * 132  # Initialize user input array with zeros
    
#     # Map symptoms fetched from JavaScript to indices in the user_input array
#     for sym in symptoms:
#         if sym in final_symptoms:
#             ind = final_symptoms.index(sym)
#             user_input[ind] = 1
    
#     user_input = np.array(user_input).reshape(1, -1)
    
#     # Use the trained classifiers to predict diseases
#     disease_nb = naive_bayes_classifier.predict(user_input)
#     disease_dt = decision_tree_classifier.predict(user_input)
#     disease_rf = random_forest_classifier.predict(user_input)
#     diseases = [disease_nb[0], disease_dt[0], disease_rf[0]]


#     # dictionary to keep count of each value
#     counts = {}
#     # iterate through the list
#     for item in diseases:
#         if item in counts:
#             counts[item] += 1
#         else:
#             counts[item] = 1
#     # get the keys with the max counts
#     disease = [key for key in counts.keys() if counts[key] == max(counts.values())]
#     response = {
#         'disease': disease[0]
#     }
        
#     return jsonify(response)  # Send the response back to the frontend




# # @app.route('/disease_description', methods=['POST'])
# @app.route('/disease_description', methods=['POST'])
# def disease_description():
#     data2 = request.json  # Get JSON data from frontend
#     disease = data2.get('disease')  # Get the disease name from JSON

#     chrome_driver_path = './chromedriver.exe'  # Path to your ChromeDriver executable
#     service = Service(chrome_driver_path)
#     options = webdriver.ChromeOptions()
#     options.add_argument('--headless')  # Run Chrome in headless mode for faster execution
#     options.add_argument('--disable-gpu')
#     options.add_argument('--no-sandbox')
    
#     driver = webdriver.Chrome(service=service, options=options)
    
#     content_html = ""
#     try:
#         # Construct the search URL using the disease name
#         search_url = f'https://www.mayoclinic.org/diseases-conditions/search-results?q={disease}'
#         driver.get(search_url)
        
#         # Wait until the element with the specified class is present on the page
#         link_element = WebDriverWait(driver, 10).until(
#             EC.presence_of_element_located((By.CLASS_NAME, 'cmp-link'))
#         )
        
#         # Extract the URL from the href attribute of the link element
#         extracted_link = link_element.get_attribute('href')
        
#         # Construct the full URL if necessary
#         full_url = f'https://www.mayoclinic.org{extracted_link}' if extracted_link.startswith('/') else extracted_link
        
#         # Navigate to the disease detail page
#         driver.get(full_url)
        
#         # Wait until the desired div element is present on the page
#         div_byline_element = WebDriverWait(driver, 10).until(
#             EC.presence_of_element_located((By.CLASS_NAME, 'phmaincontent_0_ctl01_divByLine'))
#         )
        
#         # Extract the inner HTML content of the div element
#         content_html = div_byline_element.get_attribute('outerHTML')

#     except Exception as e:
#         # Handle exceptions such as timeouts or elements not being found
#         content_html = f"An error occurred while fetching the disease description: {e}"

#     finally:
#         # Ensure that the WebDriver is properly closed
#         driver.quit()
    
#     # Prepare and return the response to send back to the frontend
#     response = {
#         'content': content_html
#     }
    
#     return jsonify(response)


# @app.route('/predict_medicine', methods=['POST'])
# def predict_medicine():
#     data2 = request.json  # Get JSON data from frontend
#     disease = data2.get('disease')  # Get symptoms array from JSON
#     # Path to your ChromeDriver executable
#     chrome_driver_path = './chromedriver.exe'
#     medicine_html = ""
#     # Create a WebDriver instance
#     service = Service(chrome_driver_path)
#     driver = webdriver.Chrome(service=service)

#     url = f"https://search.medscape.com/search/?q=%22{disease}%22&plr=ref&contenttype=Drugs+%26+Neutraceuticals&page=1"

#     try:
#         # Open the URL in the browser
#         driver.get(url)

#         # Find the divs with class "searchResult"
#         div_responsive_tables = driver.find_elements(By.CLASS_NAME, "searchResult")

#         if div_responsive_tables:
#             # Extract the HTML content of the first 5 divs
#             for index, div_responsive_table in enumerate(div_responsive_tables[:6]):
#                 div_html_content = div_responsive_table.get_attribute('outerHTML')
#                 medicine_html += div_html_content
#                 # print(f"HTML content of div {index + 1} with class='searchResult':")
#                 # print(div_html_content)
#         else:
#             print("Divs with class='searchResult' not found")

#     except Exception as e:
#         print("An error occurred:", e)

#     finally:
#         # Close the browser
#         driver.quit()

#     # Prepare the response to send back to the frontend
#     response = {
#         'medicine' : medicine_html
#     }
        
#     return jsonify(response)  # Send the response back to the frontend



# if __name__ == '__main__':
#     app.run(debug=True)


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options

# # Suppress non-critical warnings
# warnings.filterwarnings("ignore", category=UserWarning, module='selenium')

# # Configure logging
# logging.basicConfig(level=logging.ERROR,  # Only log errors to keep terminal clean
#                     format='%(asctime)s - %(levelname)s - %(message)s')

# logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load and prepare your dataset
training_data = pd.read_csv('./datasets/Training.csv')
testing_data = pd.read_csv('./datasets/Testing.csv')
training_data = training_data[training_data.columns[:-1]]
all_symptoms = training_data.columns
final_symptoms = [i.replace('_', ' ') for i in all_symptoms]
training_data.columns = final_symptoms
testing_data.columns = final_symptoms
final_symptoms = final_symptoms[:-1]

# Save all symptoms to a text file
with open('all_symptoms.txt', 'w') as output_file:
    output_file.write(', '.join(f'"{symptom}"' for symptom in final_symptoms))

# Split dataset into features (X) and target (y)
X_train = training_data.iloc[:, :132]  # Features
y_train = training_data.iloc[:, -1]    # Target
X_test = testing_data.iloc[:, :132]    # Features
y_test = testing_data.iloc[:, -1]      # Target

# Train the classifiers
naive_bayes_classifier = MultinomialNB()
decision_tree_classifier = DecisionTreeClassifier(random_state=42)
random_forest_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

naive_bayes_classifier.fit(X_train, y_train)
decision_tree_classifier.fit(X_train, y_train)
random_forest_classifier.fit(X_train, y_train)

@app.route('/predict_disease', methods=['POST'])
def predict_disease():
    data = request.json
    symptoms = data.get('symptoms', [])
    
    user_input = [0] * 132
    for sym in symptoms:
        if sym in final_symptoms:
            user_input[final_symptoms.index(sym)] = 1
    
    user_input = np.array(user_input).reshape(1, -1)
    
    # Predict using the trained models
    diseases = [
        naive_bayes_classifier.predict(user_input)[0],
        decision_tree_classifier.predict(user_input)[0],
        random_forest_classifier.predict(user_input)[0]
    ]
    
    # Return the most frequent prediction
    disease = max(set(diseases), key=diseases.count)
    return jsonify({'disease': disease})

@app.route('/disease_description', methods=['POST'])
def disease_description():
    data2 = request.json
    disease = data2.get('disease')

    chrome_driver_path = './chromedriver.exe'
    service = Service(chrome_driver_path)
    options = webdriver.ChromeOptions()
    options.add_argument('--headless')
    driver = webdriver.Chrome(service=service, options=options)
    
    paragraphs = ["anil"]
    try:
        # Navigate to the search results page
        search_url = f'https://www.mayoclinic.org/diseases-conditions/search-results?q={disease}'
        driver.get(search_url)
        
        # Wait and get the first link with class name 'cmp-link'
        link_element = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CLASS_NAME, 'cmp-link'))
        )
        
        extracted_link = link_element.get_attribute('href')
        if extracted_link:
            # Navigate to the detailed page
            driver.get(extracted_link)
            
            # Wait for content with class 'content' to load
            content_element = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CLASS_NAME, 'content'))
            )
            
            # Find the div inside content that contains the paragraphs
            div_elements = content_element.find_elements(By.XPATH, './div')
            
            if div_elements:
                # Extract paragraphs from the first div element
                div_element = div_elements[0]  # Assuming we want the first div inside content
                paragraph_elements = div_element.find_elements(By.TAG_NAME, 'p')
                
                if paragraph_elements:
                    for para in paragraph_elements[:3]:  # Get top 3 paragraphs
                        paragraphs.append(para.text)
                else:
                    paragraphs.append("No paragraphs found in the content.")
            else:
                paragraphs.append("No div elements found in the content.")
        else:
            paragraphs.append("No valid link found for the disease description.")

    except WebDriverException as e:
        logger.error(f"WebDriverException: {e}")
        paragraphs.append(f"An error occurred while fetching the disease description: {e}")

    except Exception as e:
        logger.error(f"Exception: {e}")
        paragraphs.append(f"An error occurred while fetching the disease description: {e}")

    finally:
        driver.quit()
    
    if not paragraphs:
        paragraphs.append("No content was extracted.")
    
    return jsonify({'paragraphs': paragraphs})


@app.route('/predict_medicine', methods=['POST'])
def predict_medicine():
    data2 = request.json
    disease = data2.get('disease')
    
    chrome_driver_path = './chromedriver.exe'
    service = Service(chrome_driver_path)
    driver = webdriver.Chrome(service=service)

    medicine_html = ""
    url = f"https://search.medscape.com/search/?q=%22{disease}%22&plr=ref&contenttype=Drugs+%26+Neutraceuticals&page=1"
    
    try:
        driver.get(url)
        
        div_responsive_tables = driver.find_elements(By.CLASS_NAME, "searchResult")
        
        if div_responsive_tables:
            for index, div_responsive_table in enumerate(div_responsive_tables[:6]):
                medicine_html += div_responsive_table.get_attribute('outerHTML')
        else:
            print("Divs with class='searchResult' not found")

    except Exception as e:
        print("An error occurred:", e)

    finally:
        driver.quit()

    return jsonify({'medicine': medicine_html})

if __name__ == '__main__':
    app.run(debug=True)
