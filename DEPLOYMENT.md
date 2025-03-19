# Deploying to Render

This guide will walk you through deploying the Ryzn Notes backend to Render.

## Prerequisites

1. A [Render](https://render.com/) account
2. A [Groq](https://groq.com/) API key
3. A Google Cloud account with the Text-to-Speech API enabled
4. Google Cloud credentials file (JSON)

## Deployment Steps

### 1. Prepare your Google Cloud credentials

Generate a service account key in JSON format from the Google Cloud Console:

1. Go to the [Google Cloud Console](https://console.cloud.google.com/)
2. Navigate to "IAM & Admin" > "Service Accounts"
3. Create a new service account or select an existing one
4. Go to the "Keys" tab
5. Add a new key and select JSON format
6. Download the JSON key file

### 2. Create a new Web Service in Render

1. Log in to your Render dashboard
2. Click "New" and select "Web Service"
3. Connect your GitHub repository
4. Configure the service:
   - **Name**: `ryzn-notes-api` (or your preferred name)
   - **Environment**: `Python`
   - **Build Command**: `pip install -r src/app/backend/requirements.txt`
   - **Start Command**: `./start.sh`

### 3. Add Environment Variables

In the "Environment" section, add the following:

1. Click "Add Environment Variable"
2. Add `GROQ_API_KEY` with your Groq API key
3. Add `ALLOWED_ORIGINS` with the URLs where your frontend will be hosted (comma-separated)
4. Add `RENDER=true` to indicate we're in a Render environment

### 4. Add Secret Files

1. In the "Secret Files" section of your Render dashboard
2. Click "Add Secret File"
3. For "Filename" enter: `/etc/secrets/google-credentials.json`
4. For "Contents" paste the content of your Google Cloud service account JSON file
5. Click "Save"

### 5. Deploy

1. Click "Create Web Service"
2. Render will automatically deploy your service
3. Once deployed, you'll get a URL like `https://ryzn-notes-api.onrender.com`

## Updating Your Deployment

When you push changes to your repository, Render will automatically redeploy your service.

## Troubleshooting

### Logs

Check the Render logs for any issues:

1. Go to your Web Service in the Render dashboard
2. Click on "Logs" in the left sidebar

### Common Issues

1. **Missing Dependencies**: Make sure all required packages are in `requirements.txt`
2. **Incorrect Paths**: Check that all paths in `main.py` are correct
3. **Environment Variables**: Ensure all required environment variables are set
4. **Secret Files**: Verify that your Google Cloud credentials are correctly added as a secret file
