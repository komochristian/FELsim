# Executing Nginx Setup

## Prerequisites
- [Docker](https://www.docker.com/products/docker-desktop)
- [Node.js](https://nodejs.org/)
- nginx — install (via Homebrew on macOS):
```bash
  brew install nginx
```

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
npm install
npm run build
```

## 4. Configure nginx

Download `nginx.conf` and place it in your nginx servers folder:

- **macOS:** `/opt/homebrew/etc/nginx/servers/vite-app.conf`
- **Linux:** `/etc/nginx/conf.d/vite-app.conf`

Update the `root` path to point to your `dist` folder:

```nginx
server {
    listen 5173;
    server_name 0.0.0.0;

    root /absolute/path/to/fel-app/dist;  # <-- update this
    index index.html;
...
}

# 4a. Fix Folder Permissions (macOS)

nginx needs read access to your dist folder and all its parent directories:

```bash
chmod 755 /path/to/fel-app/dist
chmod 755 /path/to/fel-app
# repeat for each parent directory up to /Users/yourusername
```

## 5. Make sure you include the path to your dist file in fel-app in root

example in nginx.conf: 

server {
    ...

    root /Users/christiankomo/Downloads/FELsim/fel-app/dist;

    ...
}

## 6. Begin nginx server with

```bash
nginx -t
sudo nginx
```

If nginx is already running:
```bash
sudo nginx -s stop
sudo nginx
```

## 7. Open server on 0.0.0.0:5173 or localhost:5173
