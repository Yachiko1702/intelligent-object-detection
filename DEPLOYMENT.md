# 🚀 GitHub Deployment Guide

This guide will help you deploy your Intelligent Object Detection System using GitHub and various cloud platforms.

## 📋 Prerequisites

1. **GitHub Account**
2. **Docker Hub Account** (for container registry)
3. **Cloud Platform Account** (Heroku, Railway, Render, etc.)
4. **Domain Name** (optional, for custom domains)

## 🛠️ Step 1: Set Up GitHub Repository

### 1.1 Initialize Git Repository
```bash
cd "c:\Users\acer\OneDrive\Desktop\Sensor 382"
git init
git add .
git commit -m "Initial commit: Enhanced Object Detection System"
```

### 1.2 Create GitHub Repository
```bash
# Create repository on GitHub first, then:
git remote add origin https://github.com/yourusername/intelligent-object-detection.git
git branch -M main
git push -u origin main
```

### 1.3 Create .gitignore
```bash
cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual Environment
venv/
env/
ENV/

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Logs
*.log
logs/

# Environment
.env
.env.local
.env.production

# Models (too large for Git)
*.pt
*.onnx
*.trt
models/

# Cache
.cache/
.pytest_cache/

# Docker
.dockerignore
EOF

git add .gitignore
git commit -m "Add gitignore"
```

## 🐳 Step 2: Set Up Docker Registry

### 2.1 GitHub Container Registry (Recommended)
```bash
# Login to GitHub Container Registry
echo ${{ secrets.GITHUB_TOKEN }} | docker login ghcr.io -u ${{ github.actor }} --password-stdin

# Tag and push manually (for testing)
docker build -t ghcr.io/yourusername/intelligent-object-detection:latest .
docker push ghcr.io/yourusername/intelligent-object-detection:latest
```

### 2.2 Docker Hub (Alternative)
```bash
# Login to Docker Hub
docker login

# Tag and push
docker build -t yourusername/object-detection:latest .
docker push yourusername/object-detection:latest
```

## 🔧 Step 3: Configure GitHub Secrets

Go to your GitHub repository → Settings → Secrets and variables → Actions

### Required Secrets:
- `GITHUB_TOKEN` (automatically available)
- `HEROKU_API_KEY`: Your Heroku API key
- `HEROKU_EMAIL`: Your Heroku email
- `HEROKU_APP_NAME`: Your Heroku app name
- `RENDER_API_KEY`: Your Render API key
- `RENDER_SERVICE_ID`: Your Render service ID
- `SLACK_WEBHOOK`: Your Slack webhook (for notifications)

### Optional Secrets:
- `DOCKER_USERNAME`: Docker Hub username
- `DOCKER_PASSWORD`: Docker Hub password
- `SENTRY_DSN`: Sentry error tracking

## ☁️ Step 4: Choose Deployment Platform

### Option A: Heroku (Free Tier Available)

1. **Install Heroku CLI**
```bash
# Windows
choco install heroku-cli
# Or download from https://devcenter.heroku.com/articles/heroku-cli
```

2. **Create Heroku App**
```bash
heroku login
heroku create your-app-name
heroku stack:set container
```

3. **Configure Environment**
```bash
heroku config:set MODEL_TYPE=detection
heroku config:set MODEL_PATH=yolov8l.pt
heroku config:set CONFIDENCE_THRESHOLD=0.25
heroku config:set INFERENCE_SIZE=1280
heroku config:set MAX_FPS=30
heroku config:set ENABLE_ADAPTIVE_THRESHOLD=true
heroku config:set ENABLE_OBJECT_GROUPING=true
heroku config:set ENABLE_COLOR_ANALYSIS=true
```

4. **Deploy**
```bash
git push heroku main
```

### Option B: Railway (Easy Setup)

1. **Install Railway CLI**
```bash
npm install -g @railway/cli
```

2. **Login and Deploy**
```bash
railway login
railway init
railway up
```

3. **Configure Environment**
```bash
railway variables set MODEL_TYPE=detection
railway variables set MODEL_PATH=yolov8l.pt
railway variables set CONFIDENCE_THRESHOLD=0.25
```

### Option C: Render (Free Tier Available)

1. **Connect GitHub to Render**
   - Go to render.com
   - Click "New Web Service"
   - Connect your GitHub repository
   - Select "Docker" environment
   - Set branch to "main"

2. **Configure Environment Variables**
   - Add all environment variables from `.env.example`
   - Set Health Check Path to `/api/stats`

3. **Deploy**
   - Click "Create Web Service"
   - Render will automatically deploy on push

### Option D: Google Cloud Run

1. **Install Google Cloud SDK**
```bash
# Windows
choco install gcloudsdk
gcloud init
```

2. **Build and Deploy**
```bash
gcloud builds submit --tag gcr.io/PROJECT-ID/object-detection
gcloud run deploy object-detection --image gcr.io/PROJECT-ID/object-detection --platform managed
```

## 🔄 Step 5: Set Up Automatic Deployment

### 5.1 GitHub Actions (Already Configured)
The `.github/workflows/deploy.yml` file includes:
- **Testing**: Automated tests on multiple Python versions
- **Building**: Docker image building and pushing
- **Security**: Vulnerability scanning with Trivy
- **Deployment**: Automatic deployment to multiple platforms
- **Notifications**: Slack notifications for deployment status

### 5.2 Manual Deployment Commands

**Heroku:**
```bash
git push heroku main
```

**Railway:**
```bash
railway up
```

**Render:**
```bash
git push origin main  # Auto-deploys if configured
```

## 📊 Step 6: Monitor Deployment

### 6.1 Health Checks
```bash
# Test health endpoint
curl https://your-app-name.herokuapp.com/api/stats
```

### 6.2 Logs Monitoring

**Heroku:**
```bash
heroku logs --tail
```

**Railway:**
```bash
railway logs
```

**Render:**
- Check dashboard for logs

### 6.3 Performance Monitoring
- Set up Sentry for error tracking
- Use Google Analytics for usage metrics
- Monitor resource usage on cloud platform

## 🔒 Step 7: Security Considerations

### 7.1 HTTPS/SSL
- All platforms provide automatic SSL
- No additional configuration needed

### 7.2 API Security
```bash
# Add rate limiting
pip install flask-limiter

# Add authentication
pip install flask-jwt-extended
```

### 7.3 Environment Variables
- Never commit `.env` files to Git
- Use platform-specific environment variable management
- Rotate API keys regularly

## 🌐 Step 8: Custom Domain (Optional)

### Heroku
```bash
heroku domains:add yourdomain.com
# Configure DNS CNAME record to your-app-name.herokuapp.com
```

### Railway
```bash
railway domains add yourdomain.com
```

### Render
- Go to dashboard → Custom Domains
- Add your domain name
- Configure DNS records

## 📈 Step 9: Scaling

### Horizontal Scaling
```bash
# Heroku
heroku ps:scale web=3

# Railway
# Upgrade plan in dashboard

# Render
# Upgrade to higher tier
```

### Vertical Scaling
- Increase memory and CPU limits
- Add GPU instances for better performance
- Use load balancers for high traffic

## 🚨 Step 10: Troubleshooting

### Common Issues:

1. **Build Failures**
   - Check Dockerfile syntax
   - Verify all dependencies in requirements.txt
   - Check GitHub Actions logs

2. **Runtime Errors**
   - Check application logs
   - Verify environment variables
   - Test locally first

3. **Performance Issues**
   - Optimize Docker image size
   - Enable GPU acceleration
   - Use CDN for static assets

4. **Memory Issues**
   - Increase memory allocation
   - Optimize model loading
   - Use model quantization

## 📞 Support

### GitHub Issues
- Create issue in repository
- Include error logs
- Describe deployment environment

### Platform Support
- Heroku: https://devcenter.heroku.com/articles/support
- Railway: https://docs.railway.app/support
- Render: https://render.com/docs/support

---

**🎉 Your Intelligent Object Detection System is now ready for production deployment!**
