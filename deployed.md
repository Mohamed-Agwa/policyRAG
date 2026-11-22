# Deployment

- **Live app:** https://nilebyte-policy-assistant.onrender.com/
- **Health check:** https://nilebyte-policy-assistant.onrender.com//health
- **Host:** Render free web service (`render.yaml`): 512 MB RAM, 0.1 CPU
- **Deploys:** triggered by GitHub Actions after tests pass on `main` (deploy hook), or manually from the Render dashboard

Note: free Render services sleep after 15 minutes without traffic. The first request after that
takes about a minute while the service wakes up. Later requests are fast.
