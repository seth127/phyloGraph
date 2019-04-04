
var x_lines;
var y_lines;
var z_lines;
var x_node;
var y_node;
var z_node;
var c_node;

function unpack(rows, key) {
    return rows.map(function(row) { return row[key]; });
}

function getNodeData(data_file) {
    Plotly.d3.csv(data_file, function(err, rows){  
     x_node = unpack(rows , 'x');
     y_node = unpack(rows , 'y');
     z_node = unpack(rows , 'z'); 
     c_node = unpack(rows , 'color');
    
    });
}

function getLineData(data_file) {
    Plotly.d3.csv(data_file, function(err, rows){
        x_lines = unpack(rows , 'x');
        y_lines = unpack(rows , 'y');
        z_lines = unpack(rows , 'z'); 
    });
}

function makePlot(node_file, line_file) {
    
    getNodeData(node_file);
    getLineData(line_file);
    
    //console.log('data/fake_points.csv')
    //console.log(x_node)
    
    var data = [{
         x: x_node,
         y: y_node,
         z: z_node,
         mode: 'markers',
         type: 'scatter3d',
         opacity: 0.95,
         marker: {
           color: c_node,
           size: 5
       }
     },  {
          type: 'scatter3d',
          mode: 'lines',
          x: x_lines,
          y: y_lines,
          z: z_lines,
          opacity: 1,
          line: {
            width: 6,
        //    color: c,
            reversescale: false
          }
        }
    ];

      var layout = {
        autosize: false,
        height: 600,
        width: 600,
        scene: {
            aspectratio: {
                x: 1,
                y: 1,
                z: 1
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

    Plotly.newPlot('graph', data, layout);

} 
 


//makePlot('data/fake_points.csv', 'data/fake_lines.csv');