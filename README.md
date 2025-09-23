# FELsim

> **⚠️ Docker currently not up to date – run the bash commands below for updated version**

---

## 🔧 Getting Started

1. Download [Docker](https://www.docker.com/).
2. Download the `docker-compose.yml` and `.env` file from the `/FELsim` directory.
3. Run the following commands in the `/FELsim` directory:

   ```bash
   docker compose pull
   docker compose up --build
   ```

4. Access the Vite app at: http://localhost:5173/

▶️ To Run the App Locally

1. Navigate to the backend directory:

```bash
cd backend/
uvicorn felAPI:app --host=0.0.0.0 --port=8000 --reload
```

2. In a new terminal, navigate to the frontend directory:

```bash
cd fel-app/
npm run dev
```

3. Access app at: http://localhost:5173/ 

