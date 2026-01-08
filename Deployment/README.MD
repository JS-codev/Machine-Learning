## How to run this deployment model:

1. Install [docker](https://docs.docker.com/desktop/setup/install/windows-install/) 
2. Under current "Deployment" folder, run the following commands:
    - `docker build -t model .`
    - `docker run -p 3000:3000 model`
3. Click on "http://localhost:3000" link to see model production in action. 

---

#### Tools utlize:
- Docker
    - Allows model to run.
- FastAPI 
    - Backend tool to allow model to sync faster when receving data.
- Flutter (Front-end) 
    - To make webpage interface to look great.
- 