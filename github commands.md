# Git and GitHub Repository Deployment Guide

Follow these sequential steps to initialize Git locally, create a remote repository on GitHub, and push your source code securely.

---

## Prerequisite: Verify Git Installation
Before beginning, ensure Git is active on your system by checking its version in your terminal (PowerShell or Bash):
```bash
git --version
```

---

## Step 1: Create a New GitHub Repository

1. Open your web browser and navigate to **[GitHub](https://github.com)**.
2. Log into your account and click the **`+` (Plus)** icon in the top-right corner, then select **New repository**.
3. Configure the following repository settings:
   * **Repository name**: `fraud-detection-platform` *(or your preferred name)*.
   * **Description**: Optional summary of your platform.
   * **Public / Private**: Select **Private** if your code contains proprietary logic or security patterns.
4. **Important**: Leave all initialization settings (**Add a README file**, **Add .gitignore**, **Choose a license**) **UNCHECKED**. Your local project folder already contains these configurations.
5. Click **Create repository**.

---

## Step 2: Initialize Git Locally and Commit Code

Open your terminal, navigate to your root project directory (`D:\fraud-detection-platform`), and execute these commands:

### 1. Initialize the Git tracking engine
```bash
git init
```

### 2. Stage all project files for tracking
This reads your local `.gitignore` file and stages all valid code assets while skipping virtual environments or temporary files.
```bash
git add .
```

### 3. Create your first permanent baseline commit
```bash
git commit -m "feat: initial commit of fraud detection platform backend and docker configurations"
```

### 4. Rename the default branch to Main
Standardizes your repository branch naming convention with GitHub.
```bash
git branch -M main
```

---

## Step 3: Link Local Project to GitHub and Push

On your newly created GitHub repository webpage, copy the remote repository URL under the **"...\...git"** format. 

### 1. Link the remote GitHub URL to your local repository
Replace `YOUR_USERNAME` and `YOUR_REPO_NAME` with your actual GitHub coordinates:
```bash
git remote add origin https://github.com/amrsaeed0092/FraudDetectoinAPIwithAWSDeployment.git
```

### 2. Verify the remote linkage is correct
```bash
git remote -v
```

### 3. Push your code to the cloud repository
```bash
git push -u origin main
```
*Note: If prompted by a pop-up window, authenticate using your GitHub account credentials or a Personal Access Token (PAT).*

---

## Summary Checklist of Daily Git Workflow Commands

Once your repository link is established, use these three commands sequentially to push updates to GitHub in the future:

```bash
git add .
git commit -m "your description of changes made"
git push
```
