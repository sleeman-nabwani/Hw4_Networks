all: simulator

simulator: simulator.py
	echo '#!/bin/bash' > simulator
	echo 'python3 simulator.py "$$@"' >> simulator
	chmod +x simulator

clean:
	rm -f simulator

.PHONY: all clean
