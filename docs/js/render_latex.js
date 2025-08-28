window.MathJax = {
  tex: {
    inlineMath: [['$', '$'], ['\\(', '\\)']],   // habilita $...$
  },
  svg: { fontCache: 'global' }
};


// Ensure all <pre> elements get the 'highlight' class for consistent styling
(function () {
  function addHighlightToPre(root) {
    var scope = root || document;
    var pres = scope.getElementsByTagName('pre');
    for (var i = 0; i < pres.length; i++) {
      var el = pres[i];
      if (!el.classList.contains('highlight')) {
        el.classList.add('highlight');
      }
    }
  }

  // Run on initial DOM load
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', function () {
      addHighlightToPre();
    });
  } else {
    addHighlightToPre();
  }

  // Re-run after each Material for MkDocs page navigation
  if (window && window.document) {
    var doc$ = window.document$;
    if (typeof doc$ !== 'undefined' && doc$ && typeof doc$.subscribe === 'function') {
      doc$.subscribe(function () {
        addHighlightToPre();
      });
    }
  }
})();
