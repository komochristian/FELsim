# FELsim


# DOCKER CURRENTLY NOT UP TO DATE, RUN COMMANDS BELOW FOR UPDATED VERSION
<span style="color: gray;">
Getting started:

1. download Docker
2. download docker-compose.yml and .env file of /FELsim
3. run 'docker compose pull' in /FELsim 
4. run 'docker compose up --build'
5. access Vite app through http://localhost:5173/
</span>

## To run:

1. cd into backend/
2. run "uvicorn apitest:app --host=0.0.0.0 --port=8000 --reload"
3. cd into fel-app/
4. run "npm run dev"
