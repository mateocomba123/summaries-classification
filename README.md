# Supervised Machine Learning
## Classification: Ticket summaries into categories


The deployed web app is live at https://summary-classifier.herokuapp.com/

Problem: tickets of a call center are manually assigned to a category-subcategory on arround 80% of the cases. There are 13 categories and arround 20 subcategories for each one category.

Idea: Based on the summary of the ticket, detect the correct category and subcategory of the ticket and assign it automatically, splitting the work in two tasks:
- First: Detect the category.
- Second: Knowing the category, detect the subcategory.

Methodology: Taking historic tickets that are well labelled, Natural Language Processing were applied to obtain the proper dataset and then, Machine Learning models were tuned.

Best model: Logistic regressions (average of 89% of accuracy to predict category + subcategory)

Conclusion: There is a huge opportunity on automate this task and prevent a lot of the wrong assignments that are currently being made, reducing the delays suffered because, for example, the ticket was assigned to an incorrect workgroup and it has to be reassigned, approved again, etc.

As part of this, for each prediction you will see the explainability of it, using Lime and eli5 libraries.

The web app was built in Python using the following libraries:
* streamlit
* pandas
* numpy
* scikit-learn
* pickle
* eli5
* nltk
* lime
