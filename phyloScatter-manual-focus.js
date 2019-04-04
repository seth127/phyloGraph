
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
//var size_node;
//var op_node;
var x_lines_f;
var y_lines_f;
var z_lines_f;
var x_node_f;
var y_node_f;
var z_node_f;
var c_node_f;


// plot the focus nodes
function getFocusNodeData(data_file, resolve, reject) {
    if (!reject) reject = (err) => console.log(err);
    return new Promise((resolve, reject) => {
        Plotly.d3.csv(data_file, function(err, rows){  
            x_node_f = unpack(rows , 'x');
            y_node_f = unpack(rows , 'y');
            z_node_f = unpack(rows , 'z'); 
            c_node_f = unpack(rows , 'color');
            resolve(true); // return true to check resolve status
        });
  });
}

function getFocusLineData(data_file, resolve, reject) {
    if (!reject) reject = (err) => console.log(err);
    return new Promise((resolve, reject) => {
    Plotly.d3.csv(data_file, function(err, rows){
            x_lines_f = unpack(rows , 'x');
            y_lines_f = unpack(rows , 'y');
            z_lines_f = unpack(rows , 'z'); 
            resolve(true); // return true to check resolve status
        });
  });
}

// plot the background nodes
function getNodeData(data_file, resolve, reject) {
    if (!reject) reject = (err) => console.log(err);
    return new Promise((resolve, reject) => {
        Plotly.d3.csv(data_file, function(err, rows){  
            x_node = unpack(rows , 'x');
            y_node = unpack(rows , 'y');
            z_node = unpack(rows , 'z'); 
            c_node = unpack(rows , 'color');
            //size_node = unpack(rows , 'size');
            //op_node = unpack(rows , 'opacity');
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
         x: x_node_f,
         y: y_node_f,
         z: z_node_f,
         mode: 'markers',
         type: 'scatter3d',
         opacity: 1.0,
         marker: {
           color: c_node_f,
           //size: size_node
           size: 15
       }
     },{
          type: 'scatter3d',
          mode: 'lines',
          x: x_lines_f,
          y: y_lines_f,
          z: z_lines_f,
          opacity: 1.0,
          line: {
            width: 5,
        //    color: c,
            reversescale: false
          }
        },{
         x: x_node,
         y: y_node,
         z: z_node,
         mode: 'markers',
         type: 'scatter3d',
         opacity: 0.3,
         marker: {
           color: c_node,
           //size: size_node
           size: 7
       }
     },  {
          type: 'scatter3d',
          mode: 'lines',
          x: x_lines,
          y: y_lines,
          z: z_lines,
          opacity: 0.3,
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

var nodeFocusPromise = getFocusNodeData('data/fake_points_focus.csv');
var lineFocusPromise = getFocusLineData('data/fake_lines_focus.csv');

var nodePromise = getNodeData('data/fake_points.csv');
var linePromise = getLineData('data/fake_lines.csv');

// check for data, then plot
Promise.all([
  nodeFocusPromise,
  lineFocusPromise,
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