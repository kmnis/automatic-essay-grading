# Automatic Essay Grading

This repositary comtains code about automatically grading the essays written by 8th-12th class students.

### Folder Structure

automatic-essay-grading
|_____ app - Essay evaluator app
|_____ data - It contains all the data files (raw and processed)
|_____ data_prep - Notebook and a corresponding python file to preprocess the data
|_____ data_viz - EDA notebooks
|_____ modeling - Data mining and ML modeling notebooks

**NOTES:**

1. Run `setup_jupyter.sh` inside a virtual environment to setup the dependencies for the project
2. To run the `Data-Mining.ipynb` notebook for topic modeling, you will have to install an additional package `bertopic`. It might create installation issues sometimes depending on your hardware and virtual environemnt. You can install it by running:

    pip install bertopic

3. To run the `Essay Evaluator` app, you'll need to have a OpenAI token in `data/openai_api.txt` file