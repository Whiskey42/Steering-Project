# Steering-Project
INTED Research Assistant Steering Project

---

## Description

This project uses the mechanistic interpretability method called "Steering", where text embeddings are "steered" in a certain direction, changing its meaning in the embedding space implemented by an LLM. The effects of steering are analyzed in different ways, both visually and numerically, as well as in written form using a generative pre-trained transformer. 

Here you will find instructions on how to clone, install and run neccessary files, a structured overview of the folders and files used, as well as in-depth descriptions on some of the specific elements. 


## Table of Contents
- [Project prerequisits](#project-prerequisits)
- [Installation](#installation)
- [Folder Structure](#folder-structure)
- [Working with Features](#working-with-features)
- [Workflow](#workflow)
- [License](#licence)

## Project prerequisits
Before installation, make sure you have:
- Python 3.10 or higher
- pip 22+
- Git
- Jupyter Notebook or Jupyterlab (for `.ipynb` notebooks)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Whiskey42/Steering-Project.git
   cd Steering-Project
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the project (see [workflow](#workflow)):
   ```bash
   python file_name.py
   ```


## Folder Structure

### `Functions/`
- **Purpose:** Contains all core Python `.py` functions used across the project.
- **You can add:** New `.py` files with reusable functions.
- **Do not:** Modify existing files unless you're updating shared logic used in notebooks.

---

### `Showcase Notebooks/`
- **Purpose:** Jupyter notebooks showcasing different analyses or results using the IMDb dataset and features.
- **You can add:** New notebooks for experiments, visualizations, or analyses.
- **Do not:** Delete or rename existing notebooks without updating references in other files.

---

### `Features/`
- **Purpose:** Contains handcrafted features and their opposites, organized by category.
- **Subfolders:**
  - `Love/`, `Norway/`, `War/` – Genre-specific features.
    - `feature.txt`: List of keywords or expressions representing the theme.
    - `opposite.txt`: Keywords representing the opposite or contrast.
- **You can add:** New categories (e.g., `Horror`, `Comedy`) with `feature.txt` and optionally `opposite.txt`.
- **Do not:** Overwrite or delete existing `.txt` files unless updating known keywords.  

See [Working with Features](#working-with-features) for details on how to add your own steering features.


---

### `imdb_top_1000_with_best_fit_genre.pkl`
- **Purpose:** A pre-processed pickle file containing the IMDb dataset with assigned best-fit genres.
- **You can add:** A new version with a version number if modifying the format.
- **Do not:** Overwrite this file without backing it up, as notebooks depend on it.

---

### `README.md`
- **Purpose:** This file — provides project context and documentation.
- **You can add:** Setup instructions, usage examples, or contribution guidelines.
- **Do not:** Leave it outdated when major changes are made.

---


## Working with Features

Each category folder in `Features/` should contain:
- `feature.txt`: Keywords representing the theme.
- `opposite.txt`: Optional, representing the contrast.

To add a new theme:
1. Create a new folder: `Features/Horror/`
2. Add your `feature.txt` and optionally `opposite.txt`.

## Workflow:
Please refer to `Showcase Notebooks/` for necessary details on how to run specific functions.
In order to start analyzing, some python files must be run in a specific order. Here is the necessary workflow:
1. `Embeddings.py` - Run this file to generate embeddings of your dataset. You may need to modify the `import_data` function in `line 31`, and the file name in `line 250`.
2. `Steering_vector.py` - Run this file to generate steering vectors. You may need to run this multiple times for different layers (change `line 252` and `line 253`) to generate the necessary steering vectors. The necessary steering vectors to run our examples are: 
- War, Layers: [10, 11]
- Love, Layers: [11]

3. You are now ready to run the other files from the `Functions` folder.

## Licence
The code in this repo is free to use. Please give us an acknowledgement if you use any of our code.

## Authors

Maintained by [@Whiskey42](https://github.com/Whiskey42) & [@minamalthes](https://github.com/minamalthes)
