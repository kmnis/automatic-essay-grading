# Automatic Essay Grading

This repositary contains code about automatically grading the essays written by 8th-12th class students.

### Project Introduction
Writing is a foundational skill, but it is difficult to hone and practice. A rapidly growing student population of students learning English as a second language (known as English Language Learners - ELLs), are especially affected by this lack of practice. Most automated evaluation is available for multiple-choice questions, but assessing short and essay answers remain a challenge. The tools that are designed for assessing writing tasks, are not designed with ELLs in mind.

The goal of this project is to assess the language proficiency of 8th-12th grade English Language Learners (ELLs). Utilizing a dataset of essays written by ELLs, we will develop a model that can power further proficiency models that better support ELLs. Using NLP and Data Mining, the resulting model could serve teachers by alleviating the grading burden and support ELLs by ensuring their work is evaluated within the context of their <u>current language level</u>.

The model will help ELLs receive more accurate feedback on their language development and expedite the grading cycle for teachers. These outcomes could enable ELLs to receive more appropriate learning tasks that will help them improve their English language proficiency by communicating their strengths and weaknesses.

### Dataset Description
The dataset comprises argumentative essays written by 8th-12th grade English Language Learners (ELLs). All essays have been pre-scored by teachers according to six analytic measures: cohesion, syntax, vocabulary, phraseology, grammar, and conventions. Each measure represents a component of proficiency in essay writing, with greater scores corresponding to greater proficiency in that measure. The scores range from 1 to 5 in increments of 0.5. Training dataset consists of 3911 essays.

The dataset was curated by Vanderbilt University and the Learning Agency Lab with support from the Bill & Melinda Gates Foundation, Schmidt Futures, and Chan Zuckerberg Initiative.

### Folder Structure
```
automatic-essay-grading
|_____ app - Essay evaluator app
|_____ data - It contains all the data files (raw and processed)
|_____ data_prep - Notebook and a corresponding python file to preprocess the data
|_____ data_viz - EDA notebooks
|_____ modeling - Data mining and ML modeling notebooks
```

**NOTES:**

1. Run `setup_jupyter.sh` inside a virtual environment to setup the dependencies for the project
2. To run the `Data-Mining.ipynb` notebook for topic modeling, you will have to install an additional package `bertopic`. It might create installation issues sometimes depending on your hardware and virtual environemnt. You can install it by running:
```
pip install bertopic
```
3. To run the `Essay Evaluator` app, you'll need to have a OpenAI token in `data/openai_api.txt` file
