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

## ▶️ To Run the App Locally

1. Make sure pip, node, and npm are installed
2. Navigate to the backend directory:

```bash
cd backend/
pip install -r requirements.txt
uvicorn felAPI:app --host=127.0.0.1 --port=8000 --reload
```

3. In a new terminal, navigate to the frontend directory:

```bash
cd fel-app/
npm install
npm run dev
```

4. Access app at: http://localhost:5173/ 

