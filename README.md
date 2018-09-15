# phyloGraph

An attempt to graph the evolution of life on this planet in three dimensions. Inspired by Planet Earth (the BBC show, and also the planet). The data for the plot is courtesy of [Tree of Life](http://tolweb.org/tree/home.pages/downloadtree.html).

## code

The real work is all done in `phyloGraph.py`. Some boilerplate to easily generate graphs is in `phyloGraph.ipynb`. The `-SANDBOX` notebooks are just that: dev sandboxes for ideas.

## To Do List

* write method to pull the data from the website instead of file
    * might need to split out part of the `load_from_raw()` file that does all the processing.
    * gonna want to hard-code a limit on number of lines
* write a wrapper to filter on an id and re-render the plot
* make the plot Z-dim by year of appearance instead of just generations
* have some smarter X- and Y-dims (maybe?)

