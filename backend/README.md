# Executing Nginx Setup

## 1. Build the Docker Image

```bash
cd backend
docker build -f Dockerfile_nginx -t nginx_demo .
```

## 2. Install Docker image and run command

```bash
docker run -p 8000:8000 -e BACKEND_API_PORT=8000 -e BACKEND_API_IP=0.0.0.0 nginx_demo
```

## 3. Run

```bash
cd ../fel-app
npm run build
```

## 4. Install nginx.conf file into nginx

example macOS

```bash
/opt/homebrew/etc/nginx/servers/nginx.conf
```

## 5. Make sure you include the path to your dist file in fel-app

## 6. Open server on 0.0.0.0:5173 or localhost:5173
