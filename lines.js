

function unpack(rows, key) {
  return rows.map(function(row) 
    { return row[key]; }); 
}

/*Plotly.d3.csv('data/fake_lines.csv', function(err, rows){
    var x = unpack(rows , 'x');
    var y = unpack(rows , 'y');
    var z = unpack(rows , 'z');
    Plotly.plot('graph', [
      {
              type: 'scatter3d',
              mode: 'lines',
              x: x,
              y: y,
              z: z
    }], {
      height: 640
    });
});*/


var x;
var y;
var z;
function getData(filename) {
    Plotly.d3.csv(filename, function(err, rows){
        x = unpack(rows , 'x');
        y = unpack(rows , 'y');
        z = unpack(rows , 'z'); 
    });
}

function makePlot() {
    console.log(x)
    Plotly.plot('graph', [
      {
              type: 'scatter3d',
              mode: 'lines',
              x: x,
              y: y,
              z: z
    }], {
      height: 640
    });
}

getData('data/fake_lines.csv');
makePlot();
