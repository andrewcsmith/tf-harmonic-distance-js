var Benchmark = require("benchmark");
var suite = new Benchmark.Suite;
suite.add("cool test", function () {
    /o/.test("Hello World!");
})
    .on('cycle', function (event) {
    var benchmark = event.target;
    console.log(benchmark.toString());
})
    .run({ async: true });
