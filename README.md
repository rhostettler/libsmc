# Sequential Monte Carlo and Markov chain Monte Carlo methods for Matlab
This is a Matlab library implementing sequential Monte Carlo (aka particle filtering and smoothing) as well as particle Markov chain Monte Carlo (PMCMC) methods. The library exclusively makes use of functional programming but makes extensive use of data structures to define models, particle systems, and parameters. The most commonly used structures are:

* `model`: Defines a probabilistic model,
* `p*`: Defines a probability density function,
* `par`: Defines a set of optional parameters for each function,
* `sys`: Contains the particle system (mostly returned from functions, but may also occasionally be used as an input).

Each of the structures is described in turn below, followed by a list of implemented algorithms and standard models included in the toolbox. The interface of each algorithm is documented in the corresponding Matlab help page.


# Data Structures
## Model Definition
The `model` structure defines a probabilistic model in terms of its state transition density, likelihood, initial state density, and possible static parameters. A model **must** include the following fields:

* `model.px0`: Initial state pdf.
* `model.px`: State transition pdf.
* `model.py`: Likelihood pdf.

Additionally, a model may include the following field:

* `model.ptheta`: **TODO: This is not quite finalized yet.**

The fields `px0`, `px`, and `py` must all be a struct of the probability density type described below. `ptheta` may **TODO: Describe**.

**TODO: Describe the general model here (github doesn't support jax, use a latex generated png instead)**

**TODO: This might be extended at some point to allow for approximate optimal proposals and such**

## Probability Densities
Probability density functions (pdfs) need to define a function to draw samples from, a function to evaluate the pdf, a function to evaluate the log-pdf (it is really the log-pdf that is used in most places), and a flag whether these functions can take whole sets of particles at once or not. The fields are:

* `rand(x, t)`: Function to draw samples `z | x` (`z` may be, e.g., the predicted state), 
* `pdf(z, x, t)`: function to evaluate the pdf of `z | x` (`z` may be, e.g., the observation `y`),
* `logpdf(z, x, t)`: function to evaluate the logarithm of the pdf (note that simply implementing `log(pdf())` is a bad idea for numerical reasons), and
* `fast`: A boolean variable which indicates whether whole particle sets can be supplied as arguments or not (to speed up computations).



The Parameter Structure
-----------------------
TODO: Describe `par`.


The Particle System
-------------------
Each function can also return the particle system as its second output variable `sys` (see above). This variable is a structure that stores all particles, their weights, etc. for all time points. The fields in this variable are may vary, but the most common ones are:

* `sys.xf`: 
* `sys.wf`: 
* `sys.r`: 
* `sys.xs`: 
* `sys.ws`: 

`sys` may also contain additional relevant fields, depending on the particular method. See the help page of each function for details (`help <function>`).

**TODO: Describe somewhere what the conventions are**

*rb_cpfas:*
* par
  * Nbar  truncation length
* sys
  * x     Nx*M*N, particles of the filter
  * w     1*M*N, weights of the filter
  * P     Nx*Nx*N, covariance matrix of the linear states


*gibbs_pmcmc:*
* par
  * Kburnin
  * Kmixing
  * sample_states
  * sample_parameters

# Algorithms




# Models







