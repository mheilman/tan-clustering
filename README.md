tan-clustering
==============

Hierarchical word clustering, following "Brown clustering" (Brown et al., 1992).

Author: Michael Heilman (mheilman@ets.org, mheilman@cs.cmu.edu)

See the docstrings in the scripts for further information.

I developed the code for Python 3.2.3, but it seems to work fine with Python 2.7 and pypy 2.0.

Usage Example
-------------

```
   python pmi_cluster.py example_input.txt example_output.txt
```
   
And then (optionally) sort by the bitstrings for easier browsing...
   
```
   sort -k 2 example_output.txt > example_output_sorted.txt
```

License
-------

This software is released under the BSD 2-clause license (http://opensource.org/licenses/BSD-2-Clause).  See LICENSE.txt.

