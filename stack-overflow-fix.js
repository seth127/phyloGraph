// helper function to unpack csv
function unpack(rows, key) {
  return rows.map(function(row) 
    { return row[key]; }); 
}


var x;
var y;
var z;

function getData(filename, resolve, reject) {
  if (!reject) reject = (err) => console.log(err);
  return new Promise((resolve, reject) => {
    Plotly.d3.csv(filename, function(err, rows) {
      x = unpack(rows, 'x');
      y = unpack(rows, 'y');
      z = unpack(rows, 'z');
      // if error
      //reject(err);
      // resolve here
      resolve(true); // return true to check resolve status
    });
  });
}

function makePlot() {
  console.log(x)
  Plotly.plot('graph', [{
    type: 'scatter3d',
    mode: 'markers',
    x: x,
    y: y,
    z: z
  }], {
    height: 640
  });
}

var promise1 = getData('data/fake_points.csv');
//var promise2 = getData('fake_data2.csv');
//var promise3 = getData('fake_data3.csv');

Promise.all([
  promise1//,promise2,promise3
]).then(function(values) {
  // check if all promises are resolved
  if (values.every(function(value) { return value == true })) {
    makePlot();
  }
})