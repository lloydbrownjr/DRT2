CXX = g++

rt: main.cc camera.h hitable.h hitable_list.h material.h ray.h sphere.h vec3.h
	$(CXX) -O3 -o rt main.cc

out.ppm: rt
	rm -f out.ppm
	time ./rt > out.ppm

out.jpg: out.ppm
	rm -f out.jpg
	ppmtojpeg out.ppm > out.jpg

clean:
	rm -f rt rt.o out.ppm out.jpg
