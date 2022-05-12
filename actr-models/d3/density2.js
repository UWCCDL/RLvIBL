// plot hist
function plotHist2(data) {
    // set the dimensions and margins of the graph
    var margin = {top: 30, right: 30, bottom: 30, left: 50},
    width = 500 - margin.left - margin.right,
    height = 300 - margin.top - margin.bottom;

    // append the svg object to the body of the page
    var svg = d3.select("#vis_hist2")
    .append("svg")
    .attr("width", width + margin.left + margin.right)
    .attr("height", height + margin.top + margin.bottom)
    .append("g")
    .attr("transform",
        "translate(" + margin.left + "," + margin.top + ")");

    // add the x Axis
    var x = d3.scaleLinear()
        .domain([-20,20])
        .range([0, width]);
    svg.append("g")
        .attr("transform", "translate(0," + height + ")")
        .call(d3.axisBottom(x));

    // add the y Axis
    var y = d3.scaleLinear()
                .range([height, 0])
                .domain([0, 0.12]);
    svg.append("g")
        .call(d3.axisLeft(y));

    // Compute kernel density estimation
    var kde = kernelDensityEstimator(kernelEpanechnikov(7), x.ticks(60))
    var density1 =  kde( data
        //.filter( function(d){return d.type === "Model 1"} )
        .map(function(d){  return d.m1; }) )
    var density2 =  kde( data
        //.filter( function(d){return d.type === "Model 2"} )
        .map(function(d){  return d.m2; }) )
        
    // Plot the area
    svg.append("path")
        .attr("class", "mypath")
        .datum(density1)
        .attr("fill", "#de2d26")
        .attr("opacity", ".3")
        .attr("stroke", "#000")
        .attr("stroke-width", 1)
        .attr("stroke-linejoin", "round")
        .attr("d",  d3.line()
            .curve(d3.curveBasis)
            .x(function(d) { return x(d[0]); })
            .y(function(d) { return y(d[1]); })
        );

    // Plot the area
    svg.append("path")
        .attr("class", "mypath")
        .datum(density2)
        .attr("fill", "#3182bd")
        .attr("opacity", ".3")
        .attr("stroke", "#000")
        .attr("stroke-width", 1)
        .attr("stroke-linejoin", "round")
        .attr("d",  d3.line()
            .curve(d3.curveBasis)
            .x(function(d) { return x(d[0]); })
            .y(function(d) { return y(d[1]); })
        );

    // Handmade legend
    svg.append("circle").attr("cx",300).attr("cy",30).attr("r", 6).style("fill", "#de2d26")  
    svg.append("circle").attr("cx",300).attr("cy",60).attr("r", 6).style("fill", "#3182bd")  
    svg.append("text").attr("x", 320).attr("y", 30).text("M1").style("font-size", "15px").attr("alignment-baseline","middle")
    svg.append("text").attr("x", 320).attr("y", 60).text("M2").style("font-size", "15px").attr("alignment-baseline","middle")
    
    // Function to compute density
    function kernelDensityEstimator(kernel, X) {
    return function(V) {
      return X.map(function(x) {
        return [x, d3.mean(V, function(v) { return kernel(x - v); })];
      });
    };
    }
  
    function kernelEpanechnikov(k) {
    return function(v) {
      return Math.abs(v /= k) <= 1 ? 0.75 * (1 - v * v) / k : 0;
    };
  }
};


