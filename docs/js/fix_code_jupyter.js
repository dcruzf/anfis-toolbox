// Normalize Jupyter code block classes to 'highlight' only
(function () {
	function fixJupyterHighlights(root) {
		var scope = root || document;
		// Select elements that have BOTH classes highlight-ipynb and hl-python
		var nodes = scope.querySelectorAll('div.highlight-ipynb.hl-python');
		nodes.forEach(function (el) {
			el.className = 'highlight';
		});
	}

	// Run on initial DOM load
	if (document.readyState === 'loading') {
		document.addEventListener('DOMContentLoaded', function () {
			fixJupyterHighlights();
		});
	} else {
		fixJupyterHighlights();
	}

	// Re-run after each Material for MkDocs page navigation
	if (window && window.document) {
		var doc$ = window.document$;
		if (typeof doc$ !== 'undefined' && doc$ && typeof doc$.subscribe === 'function') {
			doc$.subscribe(function () {
				fixJupyterHighlights();
			});
		}
	}
})();
