# Flask Chat Application

This project is a simple chat application built with Flask. It demonstrates basic Flask routing, template rendering, and the use of the OpenAI API for generating chatbot responses.

## Prerequisites

Before you begin, ensure you have installed:

- Python 3.7 or later
- [Poetry](https://python-poetry.org/docs/) for Python dependency management

## Setup

To set up the project, follow these steps:

1. **Clone the Repository**

   Clone this repository to your local machine using:

git clone <repository-url>


Navigate to the project directory:

cd <project-directory>


2. **Initialize Poetry**

Initialize the project with Poetry if you haven't done so already (this step is for project creation and may be skipped if the `pyproject.toml` exists):

poetry init --no-interaction


3. **Install Dependencies**

Install the project dependencies with Poetry:

poetry add flask


Then, install all dependencies defined in `pyproject.toml`:

poetry install


## Launching the Application

To run the Flask application:

1. **Activate the Virtual Environment**

Poetry creates and manages the virtual environment for you. To run commands within this environment, use `poetry run`.

2. **Start the Flask Server**

Use the following command to start the Flask development server:

poetry run flask run


This will start the server on `http://127.0.0.1:5000/`. Navigate to this URL in your web browser to interact with the application.

## Contributing

Contributions to this project are welcome! Please consider forking the repository and submitting a pull request with your enhancements or fixes.

## License

This project is open source and available under the [MIT License](LICENSE).