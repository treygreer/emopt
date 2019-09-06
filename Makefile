all:
	$(MAKE) -C src/

clean:
	rm -rf build
	$(MAKE) -C src/ clean



