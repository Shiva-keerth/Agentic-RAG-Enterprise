# ==============================================================================
# WHAT IS A DOCKERFILE?
# Think of this file as a "recipe" for a virtual computer.
# It tells Docker exactly what Operating System to install, what Python version
# to use, what files to copy, and what commands to run to start our app.
# By using this, our app will run EXACTLY the same on your laptop, my laptop,
# and an AWS Cloud Server. No more "it works on my machine!" errors.
# ==============================================================================

# 1. BASE IMAGE (The Foundation)
# We start by downloading a tiny, pre-configured Linux computer that already 
# has Python 3.10 installed on it. "slim" means it's a lightweight version.
FROM python:3.10-slim

# 2. SYSTEM DEPENDENCIES (The Tools)
# Our Python app uses Streamlit and ChromaDB, which require some C++ build tools
# and SQLite underlying system packages. We install them on the Linux system here.
# (apt-get is the Linux equivalent of pip, but for system software).
RUN apt-get update && apt-get install -y \
    build-essential \
    sqlite3 \
    && rm -rf /var/lib/apt/lists/*

# 3. WORKING DIRECTORY (The Folder)
# We create a folder inside the virtual Linux computer called /app.
# From this point forward, every command runs inside this /app folder.
WORKDIR /app

# 4. INSTALL PYTHON LIBRARIES (The Requirements)
# First, we copy ONLY the requirements.txt file from your Windows laptop into the Linux /app folder.
# We do this first because Docker "caches" steps. If you change your Python code (app.py), 
# Docker doesn't have to re-install all the pip packages unless requirements.txt changes!
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. COPY THE ACTUAL CODE (The App)
# Now we copy the rest of your files (app.py, rag_engine.py, etc.) into the /app folder.
COPY . .

# 6. OPEN A PORT (The Door)
# Streamlit runs on port 8501 by default. We tell the Docker container to "open" 
# this port so traffic from the outside internet (AWS) can reach the Streamlit app.
EXPOSE 8501

# 7. THE LAUNCH COMMAND (The Execution)
# When the virtual computer turns on, what command should it run?
# We tell it to run: streamlit run app.py --server.address=0.0.0.0
# (0.0.0.0 means "allow connections from anywhere on the internet").
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
