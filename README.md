# phyloGraph

An attempt to graph the evolution of life on this planet in three dimensions. Inspired by Planet Earth (the BBC show, and also the planet). The data for the plot is courtesy of [Tree of Life](http://tolweb.org/tree/home.pages/downloadtree.html).

## Command Line Use

`makePhyloGraph.py` makes a plot of the evolutionary tree, starting at the node specified at --root.

You MUST have a plotly account with the username and api key in a file
named ~/.Plotly/.Credentials for this script to work.
You can go to http://plot.ly to sign up for an account. It's free.

Run `python --help makePhyloGraph.py` for info on arguments, etc.

You can run `python searchTreeOfLife.py {search name}` to search for
a list of an organism and it's corresponding id.

## code

The real work is all done in `phyloGraph.py`. Some boilerplate to easily generate graphs is in `phyloGraph.ipynb`. The `-SANDBOX` notebooks are just that: dev sandboxes for ideas.

Some example plots are published on [my plotly account](https://plot.ly/~seth127/).

## To Do List
* have click trigger a refocus()
* try mds (on cf_text=True?) but adding some random noise to it
* adjust the log scale (add 5 or 10 before logging?)
* CHECK ON WHY Homo Sapiens (and friends) ARE AT 5MM YEARS (getting filled somehow)
* figure out how to format labels for non-focus nodes
* fix year (i.e. 2017 for Chrotomys) getting parsed into age
* put back in max_depth and max_nodes stuff
* have some smarter X- and Y-dims (maybe?)
    * would be awesome to scrape classification stuff from wikipedia and the PCA it, but having trouble getting it from infobox. 
    * Could maybe just tokenize whatever's in the infobox and PCA that.
* scrape a bunch of years and write to file so I don't have to do it every time.
* fix kinfolk color ("group") when only one class in focus selection
* fix links disappearing when you hover to click on them (might be hard)
* build a UI with flask or something
    * can plotly be embedded in a way that refreshes automatically?
* function to re-render with a specific point as the root (Ideally you could click on a point and either filter or make it the root, but I don't think plotly supports that.)
* write a wrapper to filter on an id and re-render the plot


