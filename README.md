# IMDB movie sentiment analysis
![Kaggle](https://img.shields.io/badge/Dataset-Kaggle-blue.svg) ![Python 3.8](https://img.shields.io/badge/Python-3.6-brightgreen.svg) ![NLTK](https://img.shields.io/badge/Library-NLTK-orange.svg)

This project uses IMDB 50k dataset of moview review where target is label as positive and
negative review.

The algorithm used is SGDClassifier since it has feature for further learning by 
incremental training check the incremental_training.ipynb for more information.


This repository consists of files required to deploy a ___Machine Learning Web App___ created with ___Flask___ on ___Heroku___ platform.

Check the project here:- https://imdb-review-analysis-movie.herokuapp.com/

* Alternatively, you can deploy your own copy of the app using this button:

    [![Deploy to Heroku](https://www.herokucdn.com/deploy/button.png)](https://heroku.com/deploy)

* Run this project locally
  Clone the repository and run it on Conda Environment with command 
  ```Python
  python app.py
  ```
   Copy the generated localhost URL and paste in browser.


_**----- Important Note -----**_<br />
• If you encounter this webapp as shown in the picture given below, it is occuring just because **free dynos for this particular month provided by Heroku have been completely used.** _You can access the webpage on 1st of the next month._<br />
• Sorry for the inconvenience.

![Heroku-Error](application-error-heroku.png)


