# phyloGraph

An attempt to graph the evolution of life on this planet in three dimensions. Inspired by Planet Earth (the BBC show, and also the planet). The data for the plot is courtesy of [Tree of Life](http://tolweb.org/tree/home.pages/downloadtree.html).

## code

The real work is all done in `phyloGraph.py`. Some boilerplate to easily generate graphs is in `phyloGraph.ipynb`. The `-SANDBOX` notebooks are just that: dev sandboxes for ideas.

Some example plots are published on [my plotly account](https://plot.ly/~seth127/).

## To Do List

* build a UI with flask or something
    * can plotly be embedded in a way that refreshes automatically?
* write a wrapper to filter on an id and re-render the plot
* make the plot Z-dim by year of appearance instead of just generations
* have some smarter X- and Y-dims (maybe?)

