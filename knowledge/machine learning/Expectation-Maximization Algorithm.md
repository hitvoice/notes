# Expectation-Maximization Algorithm

* E-step: “augment” data by predicting values of useful hidden variables
* M-step: exploit the “augmented data” to improve estimate of parameters (“improve” is guaranteed in terms of likelihood)

Properties:
* General algorithm for computing ML estimate of mixture models
* Hill-climbing, so can only converge to a local maximum (depending on initial points)

Examples:
* k-means, GMM
* pLSA
* LDA by variational inference
* ...

reference:
Text Mining: [https://www.coursera.org/learn/text-mining](https://www.coursera.org/learn/text-mining)