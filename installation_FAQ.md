
# Installation Issues



### typing_extensions

Some users had issues with version of typing_extensions in schrodinger environment. Solution:
```
source schrodinger.2023-3.ve/bin/activate
pip install --upgrade typing_extensions
export PYTHONPATH=/path/to/schrodinger.2023-3.ve/lib/python3.8/site-packages/:/path/to/env_edn/lib/python3.8/site-packages/
$SCHRODINGER/run python3 -m src.frag_adder.run_FRAME …
```