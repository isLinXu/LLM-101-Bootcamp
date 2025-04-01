// 先加载MathJax配置
window.MathJax = {
  tex: {
    inlineMath: [['$', '$'], ['\\(', '\\)']], // 行内公式
    displayMath: [['$$', '$$'], ['\\[', '\\]']], // 行间公式
    processEscapes: true,
    processEnvironments: true,
    tags: 'ams'
  },
  options: {
    ignoreHtmlClass: 'tex2jax_ignore',
    processHtmlClass: 'tex2jax_process'
  },
  svg: {
    fontCache: 'global'
  }
};

// 动态加载MathJax库
(function () {
  var script = document.createElement('script');
  script.src = 'https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js';
  script.async = true;
  document.head.appendChild(script);
})();