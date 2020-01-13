# lab.py

Python module with utilities written during undergraduate physics laboratory courses at University of Pisa.

Although the fitting routines written here might still be useful, I recommend
using the python module [lsqfit](https://github.com/gplepage/lsqfit).

### Functionality
* curve fitting
* digitized sample fitting
* uncertainty formatting
* automatic computing of multimeter uncertainties
* formatting a table to LaTeX

### Requirements
Requirements: **scipy**, sympy, uncertainties.

Optional requirements (needed by some functions): numdifftools, matplotlib.

The modules uncertainties and numdifftools require only numpy and no compilation.

**NOT TESTED ON PYTHON 2**, but should work.
