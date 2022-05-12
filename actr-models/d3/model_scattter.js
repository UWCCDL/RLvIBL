// function to plot chart
plotChart = function(data) {
    // console.log(data[0])
    // plot chart
    const dummy_dataset  = {
      labels: data.map(d=>d.sortedIndex),
      datasets:[{ 
        type:'scatter',
        data: data.map(d=>d.m1),
        label:"LogLikelihood of M1",   // legend name 
        fill:true,    // fill areas below line
        pointRadius: 5,
        pointHoverRadius: 15,
        backgroundColor: '#fff',
        pointBackgroundColor: 'rgba(222, 45, 38, 0.5)',
        tension:0.1,  // add curvy at edge
        order: 1
      }, {
        type:'scatter',
        data:data.map(d=>d.m2),
        label:"LogLikelihood of M2",   // legend name 
        fill:true,    // fill areas below line
        pointRadius: 5,
        pointHoverRadius: 15,
        backgroundColor: '#fff',
        pointBackgroundColor: 'rgba(49, 130, 189, 0.5)',//'#3182bd',
        tension:0.1,  // add curvy at edge
        order: 1
      }, {
        type:'bar',
        data:data.map(d=>d.diff),
        label:"LL Diff",   // legend name 
        fill:false,    // fill areas below line
        borderColor: 'rgb(255, 99, 132)',
        backgroundColor: 'rgba(255, 99, 132, 0.8)',
        //backgroundColor: '#fff',
        //pointBackgroundColor: 'yellow',
        order: 1,
        barPercentage: 0.5,
        //barThickness: 6,
        //maxBarThickness: 8,
        //minBarLength: 2,
      }]
    }
    
    // // set the dimensions and margins of the graph
    // var margin = {top: 30, right: 30, bottom: 30, left: 50},
    // width = 100 - margin.left - margin.right,
    // height = 50 - margin.top - margin.bottom;

    // append the svg object to the body of the page
    // var svg = d3.select("#vis")
    // .append("canvas")
    // .attr("width", width + margin.left + margin.right)
    // .attr("height", height + margin.top + margin.bottom)
    // .attr("id", "mychart1")
    // .append("g")
    // .attr("transform",
    //     "translate(" + margin.left + "," + margin.top + ")");
    
    const ctx1 = 'max_loglikelihood';
    const chart = new Chart(ctx1, {
      //type:"bar",
      data: dummy_dataset,
      options:{
        responsive:true,
        interaction: {
          mode: 'index',
          axis: 'y'
        },
        plugins: {
          title: {
              display: true,
              text: 'Max LogLikelihood of ACT-R Model Fitting'
          }
        },
        onClick: (e) => clickHandler(e)
        // getElementsAtEventForMode: https://www.chartjs.org/docs/latest/developers/api.html#getelementsatevente
      }
      
    });

    // add ineraction
    function clickHandler (evt) {
      const points = chart.getElementsAtEventForMode(evt, 'nearest', { intersect: true }, true);
      if (points.length) {
          const firstPoint = points[0];
          const label = chart.data.labels[firstPoint.index];
          // const value = chart.data.datasets[firstPoint.datasetIndex].data[firstPoint.index];
          currID = label;
          console.log(currID)
          // plot behav
          plotUpdateBeh(subjectChart, behData, currID);
          plotUpdateModelData(modelCharts, mData, currID)
        }
      
   }
    
  };