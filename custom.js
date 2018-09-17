var waitForPlotly = setInterval( function() {
if( typeof(window.Plotly) !== "undefined" ){
    MathJax.Hub.Config({ SVG: { font: "STIX-Web" }, displayAlign: "center" });
    MathJax.Hub.Queue(["setRenderer", MathJax.Hub, "SVG"]);
    clearInterval(waitForPlotly);
}}, 250 );