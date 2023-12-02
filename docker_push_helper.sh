#!/bin/bash

docker build -t dependencyimage -f DockerfileDepImg .
docker build -t finalimage -f FinalDockerfile . 
docker tag finalimage:latest crm22aie243.azurecr.io/finalimage:latest
docker push crm22aie243.azurecr.io/finalimage:latest
