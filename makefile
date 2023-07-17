IMAGE ?= dust-filtration:latest
RUN ?= docker run -i -t --user=$$(id -u):$(id -g)  -v$$(pwd):/work -w /work  --net=host $(IMAGE)

.PHONY: enter matlab

enter:
	$(RUN)

matlab:
	docker run -it --shm-size=512M --net=host -v$$(pwd):/home/matlab/Documents/MATLAB/ContainerFolder mathworks/matlab:r2023a -browser
