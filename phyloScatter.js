
// helper function to unpack csv
function unpack(rows, key) {
  return rows.map(function(row) 
    { return row[key]; }); 
}

var x_lines;
var y_lines;
var z_lines;
var x_node;
var y_node;
var z_node;
var c_node;
var size_node;
var op_node;


function getNodeData(data_file, resolve, reject) {
    if (!reject) reject = (err) => console.log(err);
    return new Promise((resolve, reject) => {
        Plotly.d3.csv(data_file, function(err, rows){  
            x_node = unpack(rows , 'x');
            y_node = unpack(rows , 'y');
            z_node = unpack(rows , 'z'); 
            c_node = unpack(rows , 'color');
            size_node = unpack(rows , 'size');
            op_node = unpack(rows , 'opacity');
            // if error
            //reject(err);
            // resolve here
            resolve(true); // return true to check resolve status
        });
  });
}

function getLineData(data_file, resolve, reject) {
    if (!reject) reject = (err) => console.log(err);
    return new Promise((resolve, reject) => {
    Plotly.d3.csv(data_file, function(err, rows){
            x_lines = unpack(rows , 'x');
            y_lines = unpack(rows , 'y');
            z_lines = unpack(rows , 'z'); 
            // if error
            //reject(err);
            // resolve here
            resolve(true); // return true to check resolve status
        });
  });
}



function makePlot(layout) {
  // fill plot data
  var data = [{
         x: x_node,
         y: y_node,
         z: z_node,
         mode: 'markers',
         type: 'scatter3d',
         opacity: 1.0,
         marker: {
           color: c_node,
           size: size_node
       }
     },  {
          type: 'scatter3d',
          mode: 'lines',
          x: x_lines,
          y: y_lines,
          z: z_lines,
          opacity: 1,
          line: {
            width: 2,
        //    color: c,
            reversescale: false
          }
        }
    ];
  //make plot
  Plotly.plot('graph', data, layout);
}

var nodePromise = getNodeData('data/fake_points.csv');
var linePromise = getLineData('data/fake_lines.csv');

// check for data, then plot
Promise.all([
  nodePromise,
  linePromise
]).then(function(values) {
  // check if all promises are resolved
  if (values.every(function(value) { return value == true })) {
    makePlot(layout_default);
  }
})

// the layout for the plot
var layout_default = {
        autosize: false,
        height: 700,
        width: 1200,
        scene: {
            aspectratio: {
                x: 1,
                y: 2,
                z: 1.5
            },
            camera: {
                center: {
                    x: 0,
                    y: 0,
                    z: 0
                },
                eye: {
                    x: 1.25,
                    y: 1.25,
                    z: 1.25
                },
                up: {
                    x: 0,
                    y: 0,
                    z: 1
                }
            },
            xaxis: {
                type: 'linear',
                showgrid: false,
                zeroline: false,
                title: '',
                ticks: '',
                showticklabels: false
            },
            yaxis: {
                type: 'linear',
                showgrid: false,
                zeroline: false,
                title: '',
                ticks: '',
                showticklabels: false
            },
            zaxis: {
                type: 'linear',
                showgrid: false,
                zeroline: false,
                title: '',
                ticks: '',
                showticklabels: false
            }
        },
        title: '3d point clustering'
    };