#/bin/bash
if [ $# -eq 0 ]
  then
    echo "No arguments supplied, please specified the grid2op version to push to docker"
fi
version=$1

echo "Pushing grid2ip verion "$version
#exit 1
docker build -t bdonnot/grid2op:$version .
docker push bdonnot/grid2op:$version
docker push bdonnot/grid2op:latest

