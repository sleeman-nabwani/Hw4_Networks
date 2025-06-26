all: simulator

simulator: simulator.py
	rm -f simulator
	echo '#!/bin/bash' > simulator
	echo 'python3 simulator.py "$$@"' >> simulator
	chmod +x simulator

clean:
	rm -f simulator

.PHONY: all clean
