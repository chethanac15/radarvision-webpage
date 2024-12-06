# micro-doppler-web-app

This repository hosts a web application for micro-Doppler-based target classification, focusing on differentiating between drones and birds using radar signatures. The model is built using a fine-tuned VGG19 architecture for classification.

## Tech Stack

- **Frontend**: HTML, CSS, JavaScript.
- **Backend**: Flask.
- **Machine Learning**: Python (PyTorch, Scikit-learn).
- **Visualization**: Matplotlib.
- **Deployment**:
  - **Containerization**: Docker.
  - **Application Server**: Gunicorn.
  - **Reverse Proxy**: Nginx.
  - **Cloud Platform**: AWS EC2.
  
 ## Project Structure
 ```sh

micro-doppler-web-app/
├── static/                    # Static files like css, js, images, etc
├── templates/                 # Frontend html files as templates for Flask
├── app.py                     # Main Flask App
├── venv                       # virtual env Folders
├── .gitignore
├── monitor.py                 # file to monitor errors and logs
├── requirements.txt           # flask requirements
├── .env                       # General environment variables
└── README.md                  # Backend documentation
