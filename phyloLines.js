

function unpack(rows, key) {
  return rows.map(function(row) 
    { return row[key]; }); 
}

Plotly.d3.csv('data/fake_both.csv', function(err, rows){
    var x = unpack(rows , 'x');
    var y = unpack(rows , 'y');
    var z = unpack(rows , 'z');
    var c = unpack(rows , 'c');
    var o = unpack(rows , 'o');
    var s = unpack(rows , 's');
    console.log(o)
    Plotly.plot('graph', [
      {
              type: 'scatter3d',
              mode: 'lines',
              x: x,
              y: y,
              z: z
    },{
              type: 'scatter3d',
              mode: 'markers',
              x: x,
              y: y,
              z: z,
              marker: {
                color: c,
                size: s,
                opacity: o
              }
    }
    ], layout);
});


var layout = {
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