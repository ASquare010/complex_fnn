# Startup repair before protocol freeze

The first launch.py prepare invocation exited 1 during standard-library imports,
before the stage log or protocol existed. A local operator.py shadowed Python's
standard-library operator when launching the script by path. The observed error
was ImportError: cannot import name 'deque' from partially initialized module
'collections', through subprocess -> threading -> functools -> collections ->
local operator.py -> contextlib -> collections.

No numerical test, CUDA work or optimizer update started. Inspection confirmed
protocol.json and prepare.log were absent. Renamed the implementation to codec.py
and updated its two imports. Original module contents are in operator.before.txt.
No equations, numerical tolerances, fixtures, budget or gates changed. One bounded
launch retry follows; normal source freezing occurs after this naming repair.
