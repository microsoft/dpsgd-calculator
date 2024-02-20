var beta = function (N, L, sigma, epoch) {
    // Compute Bayes security.
    p = L/N;
    T = Math.round(epoch / p);

    return 1 - math.erf((p * math.sqrt(T)) / (math.sqrt(2) * sigma));
};

var tpr_at_fpr = function(beta, fpr) {
    return math.min(1 + fpr - beta, 1);
};

bayesSecurityPlot = function () {
    // Init board.
    var max_epochs = 50;
    var board = JXG.JSXGraph.initBoard("bayesSecurityPlot", {
        axis: true,
        boundingbox: [0, 1.01, max_epochs + 1, 0.83],
        showNavigation: false,
        axis: false,
    });

    var xaxis = board.create("axis", [
        [1, 0.85],
        [max_epochs, 0.85],
    ]);
    var yaxis = board.create("axis", [
        [1, 0.8],
        [1, 0.85],
    ]);
    var xlabel = board.create("text", [15, 0.84, "Epochs"]);
    var ylabel = board.create("text", [0.55, 0.9, "Bayes security"]);
    ylabel.addRotation(90);

    // Privacy bands.
    var plot_band = function (min_beta, max_beta, color) {
        var p1 = board.create("point", [1, min_beta]);
        var p2 = board.create("point", [max_epochs + 1, min_beta]);
        var p3 = board.create("point", [max_epochs + 1, max_beta]);
        var p4 = board.create("point", [1, max_beta]);
        var pol = board.create("polygon", [p1, p2, p3, p4], {
        hasInnerPoints: true,
        fillColor: color,
        fillOpacity: 0.5,
        withLines: false,
        highlight: false,
        fixed: true,
        });

        for (let i = 0; i < pol.vertices.length - 1; i++) {
        pol.vertices[i].setAttribute({ visible: false });
        }
    };

    plot_band(0.98, 1, "green");
    plot_band(0.9, 0.98, "orange");
    plot_band(0.85, 0.9, "red");

    var plot = board.create(
        "functiongraph",
        [
        function (epoch) {
            return beta(N, L, sigma, epoch);
        },
        1,
        max_epochs,
        ],
        { strokeWidth: 3, highlight: false }
    );

    return board;
};

tprfprPlot = function () {
    // Init board.
    var board = JXG.JSXGraph.initBoard("tprfprPlot", {
      axis: true,
      boundingbox: [-0.1, 1.1, 1.1, -0.1],
      showNavigation: false,
      defaultAxes: {
        x: {
          name: "FPR",
          withLabel: true,
          label: {
            position: 'bot',
          }
        },
        y: {
          name: "TPR",
          withLabel: true,
          label: {
            position: 'top',
          }
        }
      }
    });

    // Plot function.
    var plot = board.create(
      "functiongraph",
      [
        function (fpr) {
          return tpr_at_fpr(betatprfpr, fpr);
        },
        0,
        1,
      ],
      { strokeWidth: 3, highlight: false }
    );

    return board;
};