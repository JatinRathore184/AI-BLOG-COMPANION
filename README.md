BlogCraft ✍️🤖
BlogCraft is a Streamlit application that leverages large language models (LLMs) and image generation APIs to create detailed, engaging blog posts complete with AI-generated images.

Features
Blog Post Generation: Generate blog content based on a title, keywords, and desired word count.

AI-powered Images: Automatically generate and include relevant images for your blog post using the Replicate API.

Category Prediction: The application predicts the blog's category based on the title and keywords you provide.

Export Options: Save your generated blog post as a PDF or a Microsoft Word (.docx) document.

Prerequisites
Before you get started, ensure you have the following installed:

Python 3.8 or higher

pip (Python package installer)

Installation
Clone the Repository
Start by cloning this project to your local machine:

git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name

Install Dependencies
Install all the required Python libraries using the requirements.txt file:

pip install -r requirements.txt

Download NLTK Data
Your application uses NLTK. To ensure everything runs smoothly, you may need to download the punkt resource. You can do this by running the following command in your terminal:

python -c "import nltk; nltk.download('punkt')"

Configuration
This application requires API keys for two services:

Groq: For the blog post generation (LLM).

Replicate: For the AI image generation.

Get Your API Keys

Sign up at GroqCloud to get your GROQ_API_KEY.

Sign up at Replicate to get your REPLICATE_API_TOKEN.

Create .env File
In the root directory of your project, create a new file named .env and add your API keys to it in the following format:

GROQ_API_KEY=YOUR_GROQ_API_KEY
REPLICATE_API_TOKEN=YOUR_REPLICATE_API_TOKEN

Note: The .env file should not be committed to Git, as it contains sensitive information.

Usage
Once you have completed the installation and configuration, you can run the Streamlit application from your terminal:

streamlit run app.py

The application will open in your web browser, and you can start generating blogs.