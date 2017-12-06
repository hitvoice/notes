# Probabilistic Modeling

### Probabilistic Modeling
A probabilistic model is a parametrized joint distribution over variables.
$$
P\left(\left.x_1,\dots,x_n,y_1,\dots,y_n\right|\theta\right)
$$
* Data: $x_1,x_2,\dots,x_n$
* Latent variables: $y_1,y_2,\dots,y_n$ 
* Parameter: $\theta$

**Inference**
$$P\left(\left.y_1,\dots,y_n\right | x_1,\dots,x_n,\theta\right) =
\frac{P\left(\left.x_1,\dots,x_n,y_1,\dots,y_n\right|\theta\right)}
{ P\left(\left.x_1,\dots,x_n\right|\theta\right)}$$
**Learning** 
(Maximun Likelyhood)
$$
\theta=\underset{\theta}{\operatorname {argmax}}\ P\left(\left.x_1,x_2,\dots,x_n\right|\theta\right)
$$
**Prediction**
$$
P\left(\left.x_{n+1},y_{n+1}\right|x_1,\dots,x_n,\theta\right)
$$
**Classification**
$$
\underset{c}{\operatorname {argmax}}\ P \left(\left.x_{n+1}\right|θ^c\right)
$$

### Bayesian Modeling
**Prior distribution**
$$P(\theta)$$
**Posterior distribution**
$$P\left(\left.y_1,\dots,y_n,\theta\right | x_1,\dots,x_n\right) =
\frac{P(\left.x_1,\dots,x_n,y_1,\dots,y_n\right|\theta)P(\theta)}
{ P\left(x_1,\dots,x_n\right)}$$
**Prediction**
$$
P\left(\left.x_{n+1}\right|x_1,\dots,x_n \right)=\int P(x_{n+1}|\theta)
P(\theta|x_1,\dots,x_n)\mathrm d \theta
$$
Baysian update worksheet for "rain in seatle" problem:
> You're about to get on a plane to Seattle. You want to know if you should bring an umbrella. You call 3 random friends of yours who live there and ask each independently if it's raining. Each of your friends has a 2/3 chance of telling you the truth and a 1/3 chance of messing with you by lying. All 3 friends tell you that "Yes" it is raining. What is the probability that it's actually raining in Seattle?

Given the prior P(rain)=0.1.
![bayes_rain.png](resources/bayesian.png)

### Reference:
Nonparametric Baysian Models: [http://videolectures.net/mlss09uk_teh_nbm/](http://videolectures.net/mlss09uk_teh_nbm/)
Blog: [Bayes's Theorem is not optional](http://allendowney.blogspot.com/2016/09/bayess-theorem-is-not-optional.html?utm_campaign=Revue%20newsletter&utm_medium=Newsletter&utm_source=revue)