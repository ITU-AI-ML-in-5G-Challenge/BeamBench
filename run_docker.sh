docker run --rm --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 -it --ipc=host --privileged --cap-add=SYS_ADMIN --device /dev/fuse --gpus all -v /:/workspace/  -p 52716:52716  --name beambench  beambench   bash

