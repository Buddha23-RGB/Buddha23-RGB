// Assuming data is an array of objects with properties x and y
var svg = d3.select("#chart-container").append("svg");

var bars = svg
  .selectAll("rect")
  .data(data)
  .enter()
  .append("rect")
  .attr("x", function (d) {
    return d.x;
  })
  .attr("y", function (d) {
    return svgHeight - d.y;
  })
  .attr("width", 10)
  .attr("height", function (d) {
    return d.y;
  });

bars.on("click", function (d) {
  // This function is called when a bar is clicked
  // You can update your tables here
  console.log("Bar clicked:", d);
});
